use crate::ast::{AssignOp, BinaryOp, UnaryOp};
use crate::sema::{
    type_size, FunctionId, GlobalInfo, Intrinsic, LValue, LocalId, LocalInfo, StructInfo,
    SwitchLabel, Type, TypedBlock, TypedExpr, TypedExprKind, TypedForInit, TypedFunction,
    TypedInitializer, TypedProgram, TypedStmt, Uniformity, WARP_VIDEO_BASE, WARP_VIDEO_WIDTH,
};
use crate::span::{Diagnostic, Span};
use std::collections::{HashMap, HashSet};

const ALLOCATABLE_REGS: usize = 13; // r13/r14 are ABI scratch; r15 is assembler scratch.
                                    // Begin with eight local homes (five transient registers). If actual lowering
                                    // needs more transient registers because spilled operands must be reloaded,
                                    // retry the function with one or two fewer homes. This makes spilling follow
                                    // measured peak pressure instead of imposing one pessimistic reserve globally.
const MAX_LOCAL_VECTOR_REGS: usize = 8;
// Deep expressions containing an inlined helper may need nearly the whole
// architectural file for transient values. Retry with as few as two local
// homes and spill the rest before reporting genuine temporary exhaustion.
const MIN_LOCAL_VECTOR_REGS: usize = 2;
const ALLOCATABLE_SCALAR_REGS: usize = 7; // s7 is the stack pointer.
const LANE_REG: u8 = 13;
const STACK_ADDR_REG: u8 = 14;
const STACK_POINTER: &str = "s7";
const STACK_TOP: u32 = 16_384;
const SIGN_BIT: u32 = 0x8000_0000;

pub fn generate(program: &TypedProgram) -> Result<String, Diagnostic> {
    Generator::new(program).program(program)
}

#[derive(Clone, Copy)]
struct Value {
    reg: u8,
    owned: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalStorage {
    Vector(u8),
    Scalar(u8),
    Frame(usize),
}

struct FunctionAllocation {
    storage: Vec<Option<LocalStorage>>,
    starts: Vec<Option<usize>>,
    ends: Vec<Option<usize>>,
    statement_points: HashMap<usize, usize>,
    end_point: usize,
    frame_words: usize,
    scalar_homes: usize,
    spills: usize,
    vector_home_limit: usize,
    call_live_out: HashMap<usize, HashSet<LocalId>>,
}

struct LifetimeCollector {
    next_point: usize,
    starts: Vec<Option<usize>>,
    ends: Vec<Option<usize>>,
    statement_points: HashMap<usize, usize>,
}

impl LifetimeCollector {
    fn new(local_count: usize) -> Self {
        Self {
            next_point: 1,
            starts: vec![None; local_count],
            ends: vec![None; local_count],
            statement_points: HashMap::new(),
        }
    }

    fn touch(&mut self, local: LocalId, point: usize) {
        self.starts[local] = Some(self.starts[local].map_or(point, |old| old.min(point)));
        self.ends[local] = Some(self.ends[local].map_or(point, |old| old.max(point)));
    }

    fn expr(&mut self, expr: &TypedExpr, point: usize, touched: &mut HashSet<LocalId>) {
        match &expr.kind {
            TypedExprKind::Literal(_) | TypedExprKind::StringAddress(_) => {}
            TypedExprKind::Intrinsic { args, .. } | TypedExprKind::Call { args, .. } => {
                for arg in args {
                    self.expr(arg, point, touched);
                }
            }
            TypedExprKind::Inline { statements, result } => {
                for statement in statements {
                    match statement {
                        TypedStmt::Decl { local, init, .. } => {
                            self.touch(*local, point);
                            touched.insert(*local);
                            self.initializer(init.as_ref(), point, touched);
                        }
                        TypedStmt::Expr(expr) => {
                            if let Some(expr) = expr {
                                self.expr(expr, point, touched);
                            }
                        }
                        _ => unreachable!("inline candidate contains a control statement"),
                    }
                }
                self.expr(result, point, touched);
            }
            TypedExprKind::LValue(value) | TypedExprKind::AddressOf(value) => {
                self.lvalue(value, point, touched)
            }
            TypedExprKind::Unary(_, operand) => self.expr(operand, point, touched),
            TypedExprKind::Binary { left, right, .. }
            | TypedExprKind::PointerDiff { left, right, .. } => {
                self.expr(left, point, touched);
                self.expr(right, point, touched);
            }
            TypedExprKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.expr(condition, point, touched);
                self.expr(then_expr, point, touched);
                self.expr(else_expr, point, touched);
            }
            TypedExprKind::PointerAdd { pointer, index, .. } => {
                self.expr(pointer, point, touched);
                self.expr(index, point, touched);
            }
            TypedExprKind::Assign { target, right, .. } => {
                self.lvalue(target, point, touched);
                self.expr(right, point, touched);
            }
            TypedExprKind::IncDec { target, .. } => self.lvalue(target, point, touched),
        }
    }

    fn lvalue(&mut self, value: &LValue, point: usize, touched: &mut HashSet<LocalId>) {
        match value {
            LValue::Local(local) => {
                self.touch(*local, point);
                touched.insert(*local);
            }
            LValue::Global(_) => {}
            LValue::Deref(pointer) => self.expr(pointer, point, touched),
            LValue::Index { base, index, .. } => {
                self.expr(base, point, touched);
                self.expr(index, point, touched);
            }
            LValue::Member { base, .. } => self.lvalue(base, point, touched),
        }
    }

    fn initializer(
        &mut self,
        init: Option<&TypedInitializer>,
        point: usize,
        touched: &mut HashSet<LocalId>,
    ) {
        if let Some(TypedInitializer::Expr(expr)) = init {
            self.expr(expr, point, touched);
        }
    }

    fn block(&mut self, block: &TypedBlock) -> HashSet<LocalId> {
        let mut touched = HashSet::new();
        for statement in &block.statements {
            touched.extend(self.statement(statement));
        }
        touched
    }

    fn statement(&mut self, statement: &TypedStmt) -> HashSet<LocalId> {
        let point = self.next_point;
        self.next_point += 1;
        self.statement_points
            .insert(statement as *const TypedStmt as usize, point);
        let mut touched = HashSet::new();
        match statement {
            TypedStmt::Decl { local, init, .. } => {
                self.touch(*local, point);
                touched.insert(*local);
                self.initializer(init.as_ref(), point, &mut touched);
            }
            TypedStmt::Expr(expr) | TypedStmt::Return(expr, _) => {
                if let Some(expr) = expr {
                    self.expr(expr, point, &mut touched);
                }
            }
            TypedStmt::Block(block) => touched.extend(self.block(block)),
            TypedStmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.expr(condition, point, &mut touched);
                touched.extend(self.statement(then_branch));
                if let Some(else_branch) = else_branch {
                    touched.extend(self.statement(else_branch));
                }
            }
            TypedStmt::While {
                condition, body, ..
            } => {
                self.expr(condition, point, &mut touched);
                touched.extend(self.statement(body));
                let loop_exit = self.next_point;
                for &local in &touched {
                    self.ends[local] = Some(self.ends[local].unwrap().max(loop_exit));
                }
            }
            TypedStmt::DoWhile {
                body, condition, ..
            } => {
                touched.extend(self.statement(body));
                self.expr(condition, point, &mut touched);
                let loop_exit = self.next_point;
                for &local in &touched {
                    self.ends[local] = Some(self.ends[local].unwrap().max(loop_exit));
                }
            }
            TypedStmt::For {
                init,
                condition,
                step,
                body,
                ..
            } => {
                if let Some(init) = init.as_deref() {
                    match init {
                        TypedForInit::Decl { local, init, .. } => {
                            self.touch(*local, point);
                            touched.insert(*local);
                            self.initializer(init.as_ref(), point, &mut touched);
                        }
                        TypedForInit::Expr(expr) => self.expr(expr, point, &mut touched),
                    }
                }
                if let Some(condition) = condition.as_deref() {
                    self.expr(condition, point, &mut touched);
                }
                touched.extend(self.statement(body));
                if let Some(step) = step.as_deref() {
                    self.expr(step, point, &mut touched);
                }
                let loop_exit = self.next_point;
                for &local in &touched {
                    self.ends[local] = Some(self.ends[local].unwrap().max(loop_exit));
                }
            }
            TypedStmt::Switch {
                expression, body, ..
            } => {
                self.expr(expression, point, &mut touched);
                touched.extend(self.statement(body));
            }
            TypedStmt::Case { body, .. } | TypedStmt::Default { body, .. } => {
                touched.extend(self.statement(body));
            }
            TypedStmt::Break(_) | TypedStmt::Continue(_) => {}
        }
        touched
    }
}

#[derive(Clone, Default)]
struct LiveTargets {
    break_live: Option<HashSet<LocalId>>,
    continue_live: Option<HashSet<LocalId>>,
}

struct CallLiveAnalyzer {
    call_live_out: HashMap<usize, HashSet<LocalId>>,
    switch_entries: Vec<Vec<HashSet<LocalId>>>,
}

impl CallLiveAnalyzer {
    fn function(function: &TypedFunction) -> HashMap<usize, HashSet<LocalId>> {
        let mut analyzer = Self {
            call_live_out: HashMap::new(),
            switch_entries: Vec::new(),
        };
        analyzer.block(&function.body, HashSet::new(), &LiveTargets::default());
        analyzer.call_live_out
    }

    fn union(mut left: HashSet<LocalId>, right: &HashSet<LocalId>) -> HashSet<LocalId> {
        left.extend(right.iter().copied());
        left
    }

    fn block(
        &mut self,
        block: &TypedBlock,
        mut live: HashSet<LocalId>,
        targets: &LiveTargets,
    ) -> HashSet<LocalId> {
        for statement in block.statements.iter().rev() {
            live = self.statement(statement, live, targets);
        }
        live
    }

    fn statement(
        &mut self,
        statement: &TypedStmt,
        live_after: HashSet<LocalId>,
        targets: &LiveTargets,
    ) -> HashSet<LocalId> {
        match statement {
            TypedStmt::Decl { local, init, .. } => {
                let mut before_init = live_after;
                before_init.remove(local);
                match init {
                    Some(TypedInitializer::Expr(expr)) => self.expr(expr, before_init),
                    _ => before_init,
                }
            }
            TypedStmt::Expr(expr) => expr
                .as_ref()
                .map_or(live_after.clone(), |expr| self.expr(expr, live_after)),
            TypedStmt::Return(expr, _) => expr
                .as_ref()
                .map_or_else(HashSet::new, |expr| self.expr(expr, HashSet::new())),
            TypedStmt::Block(block) => self.block(block, live_after, targets),
            TypedStmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let then_live = self.statement(then_branch, live_after.clone(), targets);
                let else_live = if let Some(branch) = else_branch {
                    self.statement(branch, live_after.clone(), targets)
                } else {
                    live_after
                };
                self.expr(condition, Self::union(then_live, &else_live))
            }
            TypedStmt::While {
                condition, body, ..
            } => {
                let mut condition_live = live_after.clone();
                loop {
                    let loop_targets = LiveTargets {
                        break_live: Some(live_after.clone()),
                        continue_live: Some(condition_live.clone()),
                    };
                    let body_live = self.statement(body, condition_live.clone(), &loop_targets);
                    let successors = Self::union(live_after.clone(), &body_live);
                    let next = self.expr(condition, successors);
                    let joined = Self::union(condition_live.clone(), &next);
                    if joined == condition_live {
                        return condition_live;
                    }
                    condition_live = joined;
                }
            }
            TypedStmt::DoWhile {
                body, condition, ..
            } => {
                let mut body_live = live_after.clone();
                loop {
                    let condition_successors = Self::union(live_after.clone(), &body_live);
                    let condition_live = self.expr(condition, condition_successors);
                    let loop_targets = LiveTargets {
                        break_live: Some(live_after.clone()),
                        continue_live: Some(condition_live.clone()),
                    };
                    let next = self.statement(body, condition_live, &loop_targets);
                    let joined = Self::union(body_live.clone(), &next);
                    if joined == body_live {
                        return body_live;
                    }
                    body_live = joined;
                }
            }
            TypedStmt::For {
                init,
                condition,
                step,
                body,
                local_ids,
                ..
            } => {
                let mut condition_live = live_after.clone();
                loop {
                    let step_live = step.as_ref().map_or_else(
                        || condition_live.clone(),
                        |step| self.expr(step, condition_live.clone()),
                    );
                    let loop_targets = LiveTargets {
                        break_live: Some(live_after.clone()),
                        continue_live: Some(step_live.clone()),
                    };
                    let body_live = self.statement(body, step_live, &loop_targets);
                    let condition_successors = if condition.is_some() {
                        Self::union(live_after.clone(), &body_live)
                    } else {
                        body_live
                    };
                    let next = condition
                        .as_ref()
                        .map_or(condition_successors.clone(), |condition| {
                            self.expr(condition, condition_successors)
                        });
                    let joined = Self::union(condition_live.clone(), &next);
                    if joined == condition_live {
                        break;
                    }
                    condition_live = joined;
                }
                let mut before = match init.as_deref() {
                    Some(TypedForInit::Decl { local, init, .. }) => {
                        let mut before_init = condition_live;
                        before_init.remove(local);
                        match init {
                            Some(TypedInitializer::Expr(expr)) => self.expr(expr, before_init),
                            _ => before_init,
                        }
                    }
                    Some(TypedForInit::Expr(expr)) => self.expr(expr, condition_live),
                    None => condition_live,
                };
                for local in local_ids {
                    before.remove(local);
                }
                before
            }
            TypedStmt::Break(_) => targets.break_live.clone().unwrap_or(live_after),
            TypedStmt::Continue(_) => targets.continue_live.clone().unwrap_or(live_after),
            TypedStmt::Switch {
                expression, body, ..
            } => {
                self.switch_entries.push(Vec::new());
                let switch_targets = LiveTargets {
                    break_live: Some(live_after.clone()),
                    continue_live: targets.continue_live.clone(),
                };
                let body_live = self.statement(body, live_after, &switch_targets);
                let entries = self.switch_entries.pop().unwrap();
                let dispatch_live = entries
                    .iter()
                    .fold(body_live, |live, entry| Self::union(live, entry));
                self.expr(expression, dispatch_live)
            }
            TypedStmt::Case { body, .. } | TypedStmt::Default { body, .. } => {
                let live = self.statement(body, live_after, targets);
                if let Some(entries) = self.switch_entries.last_mut() {
                    entries.push(live.clone());
                }
                live
            }
        }
    }

    fn expr_sequence(
        &mut self,
        expressions: &[TypedExpr],
        mut live: HashSet<LocalId>,
    ) -> HashSet<LocalId> {
        let mut prefix_dependencies = Vec::with_capacity(expressions.len());
        let mut prefix = HashSet::new();
        for expression in expressions {
            prefix_dependencies.push(prefix.clone());
            prefix.extend(value_home_dependencies(expression));
        }
        for (expression, dependencies) in expressions.iter().zip(prefix_dependencies).rev() {
            live.extend(dependencies);
            live = self.expr(expression, live);
        }
        live
    }

    fn expr(&mut self, expr: &TypedExpr, live_after: HashSet<LocalId>) -> HashSet<LocalId> {
        match &expr.kind {
            TypedExprKind::Literal(_) | TypedExprKind::StringAddress(_) => live_after,
            TypedExprKind::Intrinsic { args, .. } => self.expr_sequence(args, live_after),
            TypedExprKind::Call { args, .. } => {
                self.call_live_out
                    .entry(expr as *const TypedExpr as usize)
                    .or_default()
                    .extend(live_after.iter().copied());
                self.expr_sequence(args, live_after)
            }
            TypedExprKind::LValue(lvalue) => self.read_lvalue(lvalue, live_after),
            TypedExprKind::AddressOf(lvalue) => self.lvalue_address(lvalue, live_after),
            TypedExprKind::Unary(_, operand) => self.expr(operand, live_after),
            TypedExprKind::Binary {
                op, left, right, ..
            } if *op == BinaryOp::LogicalAnd || *op == BinaryOp::LogicalOr => {
                let right_live = self.expr(right, live_after.clone());
                self.expr(left, Self::union(live_after, &right_live))
            }
            TypedExprKind::Binary {
                op, left, right, ..
            } if *op == BinaryOp::Comma => {
                let right_live = self.expr(right, live_after);
                self.expr(left, right_live)
            }
            TypedExprKind::Binary { left, right, .. }
            | TypedExprKind::PointerDiff { left, right, .. } => {
                let mut right_after = live_after;
                right_after.extend(value_home_dependencies(left));
                let right_live = self.expr(right, right_after);
                self.expr(left, right_live)
            }
            TypedExprKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let then_live = self.expr(then_expr, live_after.clone());
                let else_live = self.expr(else_expr, live_after);
                self.expr(condition, Self::union(then_live, &else_live))
            }
            TypedExprKind::PointerAdd { pointer, index, .. } => {
                let mut index_after = live_after;
                index_after.extend(value_home_dependencies(pointer));
                let index_live = self.expr(index, index_after);
                self.expr(pointer, index_live)
            }
            TypedExprKind::Assign {
                target, op, right, ..
            } => {
                let mut right_after = live_after;
                if let LValue::Local(local) = target {
                    right_after.remove(local);
                } else {
                    right_after = self.lvalue_address(target, right_after);
                }
                if *op != AssignOp::Assign {
                    right_after = self.read_lvalue(target, right_after);
                }
                self.expr(right, right_after)
            }
            TypedExprKind::IncDec { target, .. } => self.read_lvalue(target, live_after),
            TypedExprKind::Inline { statements, result } => {
                let mut live = self.expr(result, live_after);
                for statement in statements.iter().rev() {
                    live = self.statement(statement, live, &LiveTargets::default());
                }
                live
            }
        }
    }

    fn read_lvalue(&mut self, lvalue: &LValue, mut live: HashSet<LocalId>) -> HashSet<LocalId> {
        match lvalue {
            LValue::Local(local) => {
                live.insert(*local);
                live
            }
            LValue::Global(_) => live,
            LValue::Deref(pointer) => self.expr(pointer, live),
            LValue::Index { base, index, .. } => {
                let mut index_after = live;
                index_after.extend(value_home_dependencies(base));
                let index_live = self.expr(index, index_after);
                self.expr(base, index_live)
            }
            LValue::Member { base, .. } => self.read_lvalue(base, live),
        }
    }

    fn lvalue_address(&mut self, lvalue: &LValue, live: HashSet<LocalId>) -> HashSet<LocalId> {
        match lvalue {
            LValue::Local(_) | LValue::Global(_) => live,
            LValue::Deref(pointer) => self.expr(pointer, live),
            LValue::Index { base, index, .. } => {
                let mut index_after = live;
                index_after.extend(value_home_dependencies(base));
                let index_live = self.expr(index, index_after);
                self.expr(base, index_live)
            }
            LValue::Member { base, .. } => self.lvalue_address(base, live),
        }
    }
}

fn value_home_dependencies(expr: &TypedExpr) -> HashSet<LocalId> {
    match &expr.kind {
        TypedExprKind::LValue(LValue::Local(local)) => HashSet::from([*local]),
        TypedExprKind::Unary(crate::ast::UnaryOp::Plus, operand) => {
            value_home_dependencies(operand)
        }
        TypedExprKind::Binary {
            op: BinaryOp::Comma,
            right,
            ..
        }
        | TypedExprKind::Assign { right, .. } => value_home_dependencies(right),
        TypedExprKind::Inline { result, .. } => value_home_dependencies(result),
        _ => HashSet::new(),
    }
}

fn intervals_overlap(
    left: LocalId,
    right: LocalId,
    starts: &[Option<usize>],
    ends: &[Option<usize>],
) -> bool {
    starts[left].unwrap() <= ends[right].unwrap() && starts[right].unwrap() <= ends[left].unwrap()
}

fn allocate_function(
    function: &TypedFunction,
    locals: &[LocalInfo],
    local_count: usize,
    data_words: usize,
    local_vector_regs: usize,
) -> Result<FunctionAllocation, Diagnostic> {
    let mut lifetime = LifetimeCollector::new(local_count);
    for &parameter in &function.params {
        lifetime.touch(parameter, 0);
    }
    lifetime.block(&function.body);
    let call_live_out = CallLiveAnalyzer::function(function);

    let mut function_locals: Vec<LocalId> = lifetime
        .starts
        .iter()
        .enumerate()
        .filter_map(|(local, start)| start.map(|_| local))
        .collect();
    function_locals.sort_by_key(|&local| (lifetime.starts[local].unwrap(), local));

    let mut storage = vec![None; local_count];
    let mut frame_words = function.frame_words;
    for &local in &function_locals {
        if let Some(offset) = locals[local].frame_offset {
            storage[local] = Some(LocalStorage::Frame(offset));
        }
    }

    let mut scalar_assignments: Vec<(LocalId, u8)> = Vec::new();
    let parameters: HashSet<LocalId> = function.params.iter().copied().collect();
    for &local in &function_locals {
        if storage[local].is_some()
            || parameters.contains(&local)
            || locals[local].uniformity != Uniformity::Uniform
        {
            continue;
        }
        let reg = (0..ALLOCATABLE_SCALAR_REGS).find(|&candidate| {
            scalar_assignments.iter().all(|&(other, assigned)| {
                assigned as usize != candidate
                    || !intervals_overlap(local, other, &lifetime.starts, &lifetime.ends)
            })
        });
        if let Some(reg) = reg {
            storage[local] = Some(LocalStorage::Scalar(reg as u8));
            scalar_assignments.push((local, reg as u8));
        }
    }

    let mut vector_assignments: Vec<(LocalId, u8)> = Vec::new();
    for (reg, &parameter) in function.params.iter().enumerate() {
        if storage[parameter].is_none() {
            storage[parameter] = Some(LocalStorage::Vector(reg as u8));
            vector_assignments.push((parameter, reg as u8));
        }
    }
    let mut spills = 0;
    for &local in &function_locals {
        if storage[local].is_some() {
            continue;
        }
        let reg = (0..local_vector_regs).find(|&candidate| {
            vector_assignments.iter().all(|&(other, assigned)| {
                assigned as usize != candidate
                    || !intervals_overlap(local, other, &lifetime.starts, &lifetime.ends)
            })
        });
        if let Some(reg) = reg {
            storage[local] = Some(LocalStorage::Vector(reg as u8));
            vector_assignments.push((local, reg as u8));
        } else {
            storage[local] = Some(LocalStorage::Frame(frame_words));
            frame_words += 1;
            spills += 1;
        }
    }

    if data_words + frame_words * 32 >= STACK_TOP as usize {
        return Err(Diagnostic::new(
            function.span,
            "Warp C data and register-spill frame exceed VM RAM",
        ));
    }
    Ok(FunctionAllocation {
        storage,
        starts: lifetime.starts,
        ends: lifetime.ends,
        statement_points: lifetime.statement_points,
        end_point: lifetime.next_point + 1,
        frame_words,
        scalar_homes: scalar_assignments
            .iter()
            .map(|(_, reg)| *reg as usize + 1)
            .max()
            .unwrap_or(0),
        spills,
        vector_home_limit: local_vector_regs,
        call_live_out,
    })
}

struct Generator<'a> {
    assembly: String,
    used: [bool; ALLOCATABLE_REGS],
    scalar_used: [bool; ALLOCATABLE_SCALAR_REGS],
    local_storage: Vec<Option<LocalStorage>>,
    local_starts: Vec<Option<usize>>,
    local_ends: Vec<Option<usize>>,
    local_active: Vec<bool>,
    statement_points: HashMap<usize, usize>,
    allocation_end_point: usize,
    peak_vector_regs: usize,
    allocation_scalar_homes: usize,
    allocation_spills: usize,
    allocation_vector_home_limit: usize,
    call_live_out: HashMap<usize, HashSet<LocalId>>,
    protected: [usize; ALLOCATABLE_REGS],
    next_label: usize,
    controls: Vec<ControlTarget>,
    switch_labels: Vec<Vec<(usize, String)>>,
    function_labels: Vec<String>,
    function_names: &'a [String],
    current_function: Option<FunctionId>,
    current_return_type: Type,
    current_frame_words: usize,
    main_function: FunctionId,
    locals: &'a [LocalInfo],
    globals: &'a [GlobalInfo],
    structs: &'a [StructInfo],
    data_words: usize,
    current_guard: Option<u8>,
    mask_stack: Vec<u8>,
}

struct ControlTarget {
    break_label: String,
    continue_label: Option<String>,
}

impl<'a> Generator<'a> {
    fn new(program: &'a TypedProgram) -> Self {
        Self {
            assembly: String::new(),
            used: [false; ALLOCATABLE_REGS],
            scalar_used: [false; ALLOCATABLE_SCALAR_REGS],
            local_storage: vec![None; program.locals.len()],
            local_starts: vec![None; program.locals.len()],
            local_ends: vec![None; program.locals.len()],
            local_active: vec![false; program.locals.len()],
            statement_points: HashMap::new(),
            allocation_end_point: 0,
            peak_vector_regs: 0,
            allocation_scalar_homes: 0,
            allocation_spills: 0,
            allocation_vector_home_limit: MAX_LOCAL_VECTOR_REGS,
            call_live_out: HashMap::new(),
            protected: [0; ALLOCATABLE_REGS],
            next_label: 0,
            controls: Vec::new(),
            switch_labels: Vec::new(),
            function_labels: program
                .function_names
                .iter()
                .map(|name| format!("__warpc_fn_{name}"))
                .collect(),
            function_names: &program.function_names,
            current_function: None,
            current_return_type: Type::VOID,
            current_frame_words: 0,
            main_function: program.main,
            locals: &program.locals,
            globals: &program.globals,
            structs: &program.structs,
            data_words: program.data_words.len(),
            current_guard: None,
            mask_stack: Vec::new(),
        }
    }

    fn program(mut self, program: &TypedProgram) -> Result<String, Diagnostic> {
        self.line("; generated by warpc — Warp C v0.1.4");
        self.line(".scratch r15");
        self.line("entry:");
        self.instr(&format!("LANEID r{LANE_REG}"));
        self.instr(&format!("S_MOV_I {STACK_POINTER}, {STACK_TOP}"));
        if program.data_words.iter().any(|word| *word != 0) {
            self.instr(&format!("CMP_EQ p0, r{LANE_REG}, 0"));
            for (address, word) in program.data_words.iter().copied().enumerate() {
                if word == 0 {
                    continue;
                }
                self.instr(&format!("MOV r0, {address}"));
                self.instr(&format!("MOV r1, {word}"));
                self.guarded_instr(0, false, "STORE r0, r1");
            }
        }
        self.instr(&format!("JMP {}", self.function_labels[program.main]));
        for function in &program.functions {
            self.function(function)?;
        }
        Ok(self.assembly)
    }

    fn function(&mut self, function: &TypedFunction) -> Result<(), Diagnostic> {
        let assembly_start = self.assembly.len();
        let label_start = self.next_label;
        let mut last_pressure_error = None;
        for local_vector_regs in (MIN_LOCAL_VECTOR_REGS..=MAX_LOCAL_VECTOR_REGS).rev() {
            self.assembly.truncate(assembly_start);
            self.next_label = label_start;
            match self.function_with_limit(function, local_vector_regs) {
                Ok(()) => return Ok(()),
                Err(error)
                    if error.message.contains("temporary vector-register demand")
                        && local_vector_regs > MIN_LOCAL_VECTOR_REGS =>
                {
                    last_pressure_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_pressure_error.expect("allocation retry must retain its diagnostic"))
    }

    fn function_with_limit(
        &mut self,
        function: &TypedFunction,
        local_vector_regs: usize,
    ) -> Result<(), Diagnostic> {
        let allocation = allocate_function(
            function,
            self.locals,
            self.locals.len(),
            self.data_words,
            local_vector_regs,
        )?;
        self.used.fill(false);
        self.scalar_used.fill(false);
        self.local_storage = allocation.storage;
        self.local_starts = allocation.starts;
        self.local_ends = allocation.ends;
        self.local_active.fill(false);
        self.statement_points = allocation.statement_points;
        self.allocation_end_point = allocation.end_point;
        self.peak_vector_regs = 0;
        self.allocation_scalar_homes = allocation.scalar_homes;
        self.allocation_spills = allocation.spills;
        self.allocation_vector_home_limit = allocation.vector_home_limit;
        self.call_live_out = allocation.call_live_out;
        self.protected.fill(0);
        self.controls.clear();
        self.switch_labels.clear();
        self.current_guard = None;
        self.mask_stack.clear();
        self.current_function = Some(function.id);
        self.current_return_type = function.return_type;
        self.current_frame_words = allocation.frame_words;
        self.line(&format!("{}:", self.function_labels[function.id]));
        for local in 0..self.local_storage.len() {
            if self.local_starts[local].is_none() {
                continue;
            }
            let home = match self.local_storage[local].unwrap() {
                LocalStorage::Vector(reg) => format!("r{reg}"),
                LocalStorage::Scalar(reg) => format!("s{reg}"),
                LocalStorage::Frame(offset) if self.locals[local].frame_offset.is_none() => {
                    format!("spill[{offset}]")
                }
                LocalStorage::Frame(offset) => format!("frame[{offset}]"),
            };
            self.line(&format!(
                "; local {}: {:?} -> {home}",
                self.locals[local].name, self.locals[local].uniformity
            ));
        }
        for (abi_reg, &local) in function.params.iter().enumerate() {
            self.used[abi_reg] = true;
            if matches!(self.local_storage[local], Some(LocalStorage::Vector(_))) {
                self.local_active[local] = true;
            }
        }
        self.update_peak_vector_regs();
        if self.current_frame_words != 0 {
            self.instr(&format!(
                "S_ADD_I {STACK_POINTER}, {STACK_POINTER}, -{}",
                self.current_frame_words * 32
            ));
        }
        for (abi_reg, &local) in function.params.iter().enumerate() {
            if matches!(self.local_storage[local], Some(LocalStorage::Frame(_))) {
                let address = self.local_address(local, function.span)?;
                self.instr(&format!("STORE r{}, r{abi_reg}", address.reg));
                self.release(address);
                self.free_reg(abi_reg as u8);
            }
        }
        self.block(&function.body)?;
        if function.id == self.main_function {
            self.instr("MOV r0, 0");
            self.emit_return();
        } else {
            if function.return_type != Type::VOID {
                self.instr("MOV r0, 0");
            }
            self.emit_return();
        }
        self.advance_allocation(self.allocation_end_point)?;
        self.line(&format!(
            "; allocation {}: vector_peak={}/{} vector_homes={} scalar_homes={} spills={} frame_words={}",
            function.name,
            self.peak_vector_regs,
            ALLOCATABLE_REGS,
            self.allocation_vector_home_limit,
            self.allocation_scalar_homes,
            self.allocation_spills,
            self.current_frame_words
        ));
        self.current_function = None;
        Ok(())
    }

    fn block(&mut self, block: &TypedBlock) -> Result<(), Diagnostic> {
        for statement in &block.statements {
            self.statement(statement)?;
        }
        Ok(())
    }

    fn statement(&mut self, statement: &TypedStmt) -> Result<(), Diagnostic> {
        let point = *self
            .statement_points
            .get(&(statement as *const TypedStmt as usize))
            .ok_or_else(|| {
                Diagnostic::new(
                    statement_span(statement),
                    "internal error: missing liveness point",
                )
            })?;
        self.advance_allocation(point)?;
        match statement {
            TypedStmt::Decl { local, init, span } => {
                self.declaration(*local, init.as_ref(), *span)?;
            }
            TypedStmt::Expr(value) => {
                if let Some(value) = value {
                    let result = self.expr(value)?;
                    self.release(result);
                }
            }
            TypedStmt::Return(value, span) => {
                if let Some(value) = value {
                    let result = self.expr(value)?;
                    if result.reg != 0 {
                        self.instr(&format!("MOV r0, r{}", result.reg));
                    }
                    self.release(result);
                } else if self.current_return_type != Type::VOID {
                    return Err(Diagnostic::new(
                        *span,
                        "internal error: non-void return has no value",
                    ));
                }
                self.emit_return();
            }
            TypedStmt::Block(block) => self.block(block)?,
            TypedStmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => self.if_statement(condition, then_branch, else_branch.as_deref())?,
            TypedStmt::While {
                condition, body, ..
            } => self.while_statement(condition, body)?,
            TypedStmt::DoWhile {
                body, condition, ..
            } => self.do_while_statement(body, condition)?,
            TypedStmt::For {
                init,
                condition,
                step,
                body,
                local_ids,
                ..
            } => self.for_statement(
                init.as_deref(),
                condition.as_deref(),
                step.as_deref(),
                body,
                local_ids,
            )?,
            TypedStmt::Break(span) => {
                let Some(target) = self.controls.last() else {
                    return Err(Diagnostic::new(
                        *span,
                        "internal error: break target missing",
                    ));
                };
                self.instr(&format!("JMP {}", target.break_label));
            }
            TypedStmt::Continue(span) => {
                let Some(label) = self
                    .controls
                    .iter()
                    .rev()
                    .find_map(|target| target.continue_label.clone())
                else {
                    return Err(Diagnostic::new(
                        *span,
                        "internal error: continue target missing",
                    ));
                };
                self.instr(&format!("JMP {label}"));
            }
            TypedStmt::Switch {
                expression,
                body,
                labels,
                ..
            } => self.switch_statement(expression, body, labels)?,
            TypedStmt::Case { id, body, span } | TypedStmt::Default { id, body, span } => {
                let Some(label) = self.switch_label(*id) else {
                    return Err(Diagnostic::new(
                        *span,
                        "internal error: switch label missing",
                    ));
                };
                self.line(&format!("{label}:"));
                self.statement(body)?;
            }
        }
        Ok(())
    }

    fn if_statement(
        &mut self,
        condition: &TypedExpr,
        then_branch: &TypedStmt,
        else_branch: Option<&TypedStmt>,
    ) -> Result<(), Diagnostic> {
        if condition.uniformity == Uniformity::Divergent || self.current_guard.is_some() {
            return self.masked_if_statement(condition, then_branch, else_branch);
        }
        let otherwise = self.label("if_else");
        let end = self.label("if_end");
        self.jump_if_zero(condition, &otherwise)?;
        self.statement(then_branch)?;
        if let Some(else_branch) = else_branch {
            self.instr(&format!("JMP {end}"));
            self.line(&format!("{otherwise}:"));
            self.statement(else_branch)?;
            self.line(&format!("{end}:"));
        } else {
            self.line(&format!("{otherwise}:"));
        }
        Ok(())
    }

    fn masked_if_statement(
        &mut self,
        condition: &TypedExpr,
        then_branch: &TypedStmt,
        else_branch: Option<&TypedStmt>,
    ) -> Result<(), Diagnostic> {
        let mask = match self.mask_stack.len() {
            0 => 3,
            1 => 2,
            _ => {
                return Err(Diagnostic::new(
                    condition.span,
                    "divergent if nesting exhausted predicate masks",
                ))
            }
        };
        let parent = self.current_guard;
        let value = self.expr(condition)?;
        if let Some(parent) = parent {
            self.raw_instr(&format!("BALLOT p0, r{}", value.reg));
            self.raw_instr(&format!("ANDMASK p{mask}, p{parent}, p0"));
        } else {
            self.raw_instr(&format!("BALLOT p{mask}, r{}", value.reg));
        }
        self.release(value);

        self.mask_stack.push(mask);
        self.current_guard = Some(mask);
        self.statement(then_branch)?;

        if let Some(else_branch) = else_branch {
            if let Some(parent) = parent {
                self.raw_instr(&format!("NOTMASK p0, p{mask}"));
                self.raw_instr(&format!("ANDMASK p{mask}, p{parent}, p0"));
            } else {
                self.raw_instr(&format!("NOTMASK p{mask}, p{mask}"));
            }
            self.statement(else_branch)?;
        }

        self.mask_stack.pop();
        self.current_guard = parent;
        Ok(())
    }

    fn while_statement(
        &mut self,
        condition: &TypedExpr,
        body: &TypedStmt,
    ) -> Result<(), Diagnostic> {
        let condition_label = self.label("while_condition");
        let end = self.label("while_end");
        self.line(&format!("{condition_label}:"));
        self.jump_if_zero(condition, &end)?;
        self.controls.push(ControlTarget {
            break_label: end.clone(),
            continue_label: Some(condition_label.clone()),
        });
        self.statement(body)?;
        self.controls.pop();
        self.instr(&format!("JMP {condition_label}"));
        self.line(&format!("{end}:"));
        Ok(())
    }

    fn do_while_statement(
        &mut self,
        body: &TypedStmt,
        condition: &TypedExpr,
    ) -> Result<(), Diagnostic> {
        let body_label = self.label("do_body");
        let condition_label = self.label("do_condition");
        let end = self.label("do_end");
        self.line(&format!("{body_label}:"));
        self.controls.push(ControlTarget {
            break_label: end.clone(),
            continue_label: Some(condition_label.clone()),
        });
        self.statement(body)?;
        self.controls.pop();
        self.line(&format!("{condition_label}:"));
        self.jump_if_nonzero(condition, &body_label)?;
        self.line(&format!("{end}:"));
        Ok(())
    }

    fn for_statement(
        &mut self,
        init: Option<&TypedForInit>,
        condition: Option<&TypedExpr>,
        step: Option<&TypedExpr>,
        body: &TypedStmt,
        local_ids: &[LocalId],
    ) -> Result<(), Diagnostic> {
        if let Some(init) = init {
            match init {
                TypedForInit::Decl { local, init, span } => {
                    self.declaration(*local, init.as_ref(), *span)?;
                }
                TypedForInit::Expr(expr) => {
                    let value = self.expr(expr)?;
                    self.release(value);
                }
            }
        }
        let condition_label = self.label("for_condition");
        let step_label = self.label("for_step");
        let end = self.label("for_end");
        self.line(&format!("{condition_label}:"));
        if let Some(condition) = condition {
            self.jump_if_zero(condition, &end)?;
        }
        self.controls.push(ControlTarget {
            break_label: end.clone(),
            continue_label: Some(step_label.clone()),
        });
        self.statement(body)?;
        self.controls.pop();
        self.line(&format!("{step_label}:"));
        if let Some(step) = step {
            let value = self.expr(step)?;
            self.release(value);
        }
        self.instr(&format!("JMP {condition_label}"));
        self.line(&format!("{end}:"));
        let _ = local_ids;
        Ok(())
    }

    fn switch_statement(
        &mut self,
        expression: &TypedExpr,
        body: &TypedStmt,
        labels: &[SwitchLabel],
    ) -> Result<(), Diagnostic> {
        let end = self.label("switch_end");
        let mapped: Vec<(usize, String)> = labels
            .iter()
            .map(|label| (label.id, self.label("switch_case")))
            .collect();
        let value = self.expr(expression)?;
        for label in labels.iter().filter(|label| label.value.is_some()) {
            let target = mapped
                .iter()
                .find(|(id, _)| *id == label.id)
                .unwrap()
                .1
                .clone();
            self.instr(&format!(
                "CMP_EQ p0, r{}, {}",
                value.reg,
                label.value.unwrap()
            ));
            self.instr(&format!("JMP_IF_ANY p0, {target}"));
        }
        let default = labels.iter().find(|label| label.value.is_none());
        if let Some(default) = default {
            let target = mapped
                .iter()
                .find(|(id, _)| *id == default.id)
                .unwrap()
                .1
                .clone();
            self.instr(&format!("JMP {target}"));
        } else {
            self.instr(&format!("JMP {end}"));
        }
        self.release(value);

        self.controls.push(ControlTarget {
            break_label: end.clone(),
            continue_label: None,
        });
        self.switch_labels.push(mapped);
        self.statement(body)?;
        self.switch_labels.pop();
        self.controls.pop();
        self.line(&format!("{end}:"));
        Ok(())
    }

    fn declaration(
        &mut self,
        local: LocalId,
        init: Option<&TypedInitializer>,
        span: Span,
    ) -> Result<(), Diagnostic> {
        if matches!(self.local_storage[local], Some(LocalStorage::Frame(_))) {
            let size = type_size(self.locals[local].ty, self.structs);
            match init {
                Some(TypedInitializer::Expr(init)) => {
                    let value = self.expr(init)?;
                    let address = self.local_address(local, span)?;
                    self.instr(&format!("STORE r{}, r{}", address.reg, value.reg));
                    self.release(address);
                    self.release(value);
                }
                Some(TypedInitializer::Words(words)) => {
                    for index in 0..size {
                        let address = self.local_address_offset(local, index, span)?;
                        let value = self.alloc_reg(span)?;
                        let word = words.get(index).copied().unwrap_or(0);
                        self.instr(&format!("MOV r{value}, {word}"));
                        self.instr(&format!("STORE r{}, r{value}", address.reg));
                        self.free_reg(value);
                        self.release(address);
                    }
                }
                None => {
                    let zero = self.alloc_reg(span)?;
                    self.instr(&format!("MOV r{zero}, 0"));
                    for index in 0..size {
                        let address = self.local_address_offset(local, index, span)?;
                        self.instr(&format!("STORE r{}, r{zero}", address.reg));
                        self.release(address);
                    }
                    self.free_reg(zero);
                }
            }
            return Ok(());
        }

        let storage = self.local_storage[local]
            .ok_or_else(|| Diagnostic::new(span, "internal error: local has no allocation"))?;
        if let Some(TypedInitializer::Expr(init)) = init {
            let value = self.expr(init)?;
            self.store_local(local, value.reg, span)?;
            self.release(value);
        } else if matches!(init, Some(TypedInitializer::Words(_))) {
            return Err(Diagnostic::new(
                span,
                "internal error: aggregate initializer assigned to register local",
            ));
        } else {
            match storage {
                LocalStorage::Vector(reg) => self.instr(&format!("MOV r{reg}, 0")),
                LocalStorage::Scalar(reg) => self.instr(&format!("S_MOV_I s{reg}, 0")),
                LocalStorage::Frame(_) => unreachable!(),
            }
        }
        Ok(())
    }

    fn jump_if_zero(&mut self, condition: &TypedExpr, label: &str) -> Result<(), Diagnostic> {
        let value = self.expr(condition)?;
        self.instr(&format!("CMP_EQ p0, r{}, 0", value.reg));
        self.release(value);
        self.instr(&format!("JMP_IF_ANY p0, {label}"));
        Ok(())
    }

    fn jump_if_nonzero(&mut self, condition: &TypedExpr, label: &str) -> Result<(), Diagnostic> {
        let value = self.expr(condition)?;
        self.instr(&format!("CMP_NE p0, r{}, 0", value.reg));
        self.release(value);
        self.instr(&format!("JMP_IF_ANY p0, {label}"));
        Ok(())
    }

    fn switch_label(&self, id: usize) -> Option<String> {
        self.switch_labels
            .iter()
            .rev()
            .find_map(|labels| labels.iter().find(|(candidate, _)| *candidate == id))
            .map(|(_, label)| label.clone())
    }

    fn expr(&mut self, expr: &TypedExpr) -> Result<Value, Diagnostic> {
        match &expr.kind {
            TypedExprKind::Literal(value) => {
                let reg = self.alloc_reg(expr.span)?;
                self.instr(&format!("MOV r{reg}, {value}"));
                Ok(Value { reg, owned: true })
            }
            TypedExprKind::StringAddress(address) => {
                let reg = self.alloc_reg(expr.span)?;
                self.instr(&format!("MOV r{reg}, {address}"));
                Ok(Value { reg, owned: true })
            }
            TypedExprKind::Intrinsic { intrinsic, args } => {
                self.intrinsic(*intrinsic, args, expr.span)
            }
            TypedExprKind::LValue(lvalue) => {
                if expr.ty.is_array() || expr.ty.is_struct() {
                    self.lvalue_address(lvalue, expr.span)
                } else {
                    self.load_lvalue(lvalue, expr.span)
                }
            }
            TypedExprKind::AddressOf(lvalue) => self.lvalue_address(lvalue, expr.span),
            TypedExprKind::Unary(op, operand) => self.unary(*op, operand, expr),
            TypedExprKind::Binary {
                op,
                left,
                right,
                operand_type,
            } => self.binary(*op, left, right, *operand_type, expr),
            TypedExprKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => self.conditional(condition, then_expr, else_expr, expr),
            TypedExprKind::PointerAdd {
                pointer,
                index,
                scale,
                subtract,
            } => {
                let pointer = self.expr(pointer)?;
                self.protect(pointer);
                let index = self.expr(index)?;
                self.unprotect(pointer);
                let scaled = if *scale == 1 {
                    index
                } else {
                    let out = self.alloc_reg(expr.span)?;
                    self.instr(&format!("MUL r{out}, r{}, {scale}", index.reg));
                    self.release(index);
                    Value {
                        reg: out,
                        owned: true,
                    }
                };
                let out = self.alloc_reg(expr.span)?;
                let mnemonic = if *subtract { "SUB" } else { "ADD" };
                self.instr(&format!(
                    "{mnemonic} r{out}, r{}, r{}",
                    pointer.reg, scaled.reg
                ));
                self.release(pointer);
                self.release(scaled);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            TypedExprKind::PointerDiff { left, right, scale } => {
                let left = self.expr(left)?;
                self.protect(left);
                let right = self.expr(right)?;
                self.unprotect(left);
                let delta = self.alloc_reg(expr.span)?;
                self.instr(&format!("SUB r{delta}, r{}, r{}", left.reg, right.reg));
                self.release(left);
                self.release(right);
                if *scale == 1 {
                    Ok(Value {
                        reg: delta,
                        owned: true,
                    })
                } else {
                    let divisor = self.alloc_reg(expr.span)?;
                    self.instr(&format!("MOV r{divisor}, {scale}"));
                    let out = self.signed_div_mod(BinaryOp::Div, delta, divisor, expr.span)?;
                    self.free_reg(divisor);
                    self.free_reg(delta);
                    Ok(Value {
                        reg: out,
                        owned: true,
                    })
                }
            }
            TypedExprKind::Assign {
                target,
                op,
                right,
                operation_type,
                scale,
            } => self.assignment(target, *op, right, *operation_type, *scale, expr),
            TypedExprKind::Call { function, args } => self.call(
                *function,
                args,
                expr.ty,
                expr.span,
                expr as *const TypedExpr as usize,
            ),
            TypedExprKind::Inline { statements, result } => {
                for statement in statements {
                    match statement {
                        TypedStmt::Decl { local, init, span } => {
                            self.declaration(*local, init.as_ref(), *span)?;
                        }
                        TypedStmt::Expr(expr) => {
                            if let Some(expr) = expr {
                                let value = self.expr(expr)?;
                                self.release(value);
                            }
                        }
                        _ => unreachable!("inline candidate contains a control statement"),
                    }
                }
                self.expr(result)
            }
            TypedExprKind::IncDec {
                target,
                increment,
                postfix,
                scale,
            } => {
                let current = self.load_lvalue(target, expr.span)?;
                if *postfix {
                    let old = self.alloc_reg(expr.span)?;
                    self.instr(&format!("MOV r{old}, r{}", current.reg));
                    let mnemonic = if *increment { "ADD" } else { "SUB" };
                    let updated = self.alloc_reg(expr.span)?;
                    self.instr(&format!("{mnemonic} r{updated}, r{}, {scale}", current.reg));
                    let old_value = Value {
                        reg: old,
                        owned: true,
                    };
                    let updated_value = Value {
                        reg: updated,
                        owned: true,
                    };
                    self.protect(old_value);
                    self.protect(updated_value);
                    self.store_lvalue(target, updated, expr.span)?;
                    self.unprotect(updated_value);
                    self.unprotect(old_value);
                    self.free_reg(updated);
                    self.release(current);
                    Ok(old_value)
                } else {
                    let mnemonic = if *increment { "ADD" } else { "SUB" };
                    let updated = self.alloc_reg(expr.span)?;
                    self.instr(&format!("{mnemonic} r{updated}, r{}, {scale}", current.reg));
                    let updated_value = Value {
                        reg: updated,
                        owned: true,
                    };
                    self.protect(updated_value);
                    self.store_lvalue(target, updated, expr.span)?;
                    self.unprotect(updated_value);
                    self.release(current);
                    Ok(updated_value)
                }
            }
        }
    }

    fn intrinsic(
        &mut self,
        intrinsic: Intrinsic,
        args: &[TypedExpr],
        span: Span,
    ) -> Result<Value, Diagnostic> {
        match intrinsic {
            Intrinsic::LaneId => Ok(Value {
                reg: LANE_REG,
                owned: false,
            }),
            Intrinsic::VmId => {
                let reg = self.alloc_reg(span)?;
                self.instr(&format!("VMID r{reg}"));
                Ok(Value { reg, owned: true })
            }
            Intrinsic::Framebuffer => {
                let reg = self.alloc_reg(span)?;
                self.instr(&format!("MOV r{reg}, {WARP_VIDEO_BASE}"));
                Ok(Value { reg, owned: true })
            }
            Intrinsic::Flip => {
                // FLIP has VM-wide semantics and cannot carry a lane predicate.
                // Semantic analysis guarantees this is outside divergence.
                self.raw_instr("FLIP");
                Ok(Value {
                    reg: 0,
                    owned: false,
                })
            }
            Intrinsic::Argb => {
                let values = self.expr_values(args)?;
                let out = self.alloc_reg(span)?;
                self.instr(&format!("AND r{out}, r{}, 255", values[0].reg));
                self.release(values[0]);
                self.instr(&format!("SHL r{out}, r{out}, 24"));

                let component = self.alloc_reg(span)?;
                for (value, shift) in values[1..].iter().copied().zip([16, 8, 0]) {
                    self.instr(&format!("AND r{component}, r{}, 255", value.reg));
                    self.release(value);
                    if shift != 0 {
                        self.instr(&format!("SHL r{component}, r{component}, {shift}"));
                    }
                    self.instr(&format!("OR r{out}, r{out}, r{component}"));
                }
                self.free_reg(component);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            Intrinsic::SetPixel => {
                let values = self.expr_values(args)?;
                let [x, y, colour] = values.as_slice() else {
                    unreachable!("set_pixel arity checked by semantic analysis")
                };
                let (x, y, colour) = (*x, *y, *colour);
                let address = self.alloc_reg(span)?;
                self.instr(&format!("MUL r{address}, r{}, {WARP_VIDEO_WIDTH}", y.reg));
                self.release(y);
                self.instr(&format!("ADD r{address}, r{address}, r{}", x.reg));
                self.release(x);
                self.instr(&format!("ADD r{address}, r{address}, {WARP_VIDEO_BASE}"));
                self.instr(&format!("STORE r{address}, r{}", colour.reg));
                self.release(colour);
                self.free_reg(address);
                Ok(Value {
                    reg: 0,
                    owned: false,
                })
            }
            Intrinsic::Send => {
                let values = self.expr_values(args)?;
                let [destination, message_type, payload] = values.as_slice() else {
                    unreachable!("warp_send arity checked by semantic analysis")
                };
                let (destination, message_type, payload) = (*destination, *message_type, *payload);
                self.instr(&format!(
                    "SEND r{}, r{}, r{}",
                    destination.reg, message_type.reg, payload.reg
                ));
                self.release(destination);
                self.release(message_type);
                self.release(payload);
                Ok(Value {
                    reg: 0,
                    owned: false,
                })
            }
            Intrinsic::TryRecv => {
                let values = self.expr_values(args)?;
                let [payload_address, metadata_address] = values.as_slice() else {
                    unreachable!("warp_try_recv arity checked by semantic analysis")
                };
                let (payload_address, metadata_address) = (*payload_address, *metadata_address);
                let payload = self.alloc_reg(span)?;
                let metadata = self.alloc_reg(span)?;
                let received = self.alloc_reg(span)?;
                self.instr(&format!("MOV r{payload}, 0"));
                self.instr(&format!("MOV r{metadata}, 0"));
                self.instr(&format!("TRY_RECV p0, r{payload}, r{metadata}"));
                self.instr(&format!("MOV r{received}, 0"));
                self.guarded_instr(0, false, &format!("MOV r{received}, 1"));
                self.guarded_instr(
                    0,
                    false,
                    &format!("STORE r{}, r{payload}", payload_address.reg),
                );
                self.guarded_instr(
                    0,
                    false,
                    &format!("STORE r{}, r{metadata}", metadata_address.reg),
                );
                self.release(payload_address);
                self.release(metadata_address);
                self.free_reg(payload);
                self.free_reg(metadata);
                Ok(Value {
                    reg: received,
                    owned: true,
                })
            }
            Intrinsic::Broadcast => {
                let value = self.expr(&args[0])?;
                let out = self.alloc_reg(span)?;
                if let TypedExprKind::Literal(lane) = args[1].kind {
                    self.instr(&format!("BROADCAST r{out}, r{}, {lane}", value.reg));
                } else {
                    self.protect(value);
                    let lane = self.expr(&args[1])?;
                    self.unprotect(value);
                    self.instr(&format!("SHUFFLE r{out}, r{}, r{}", value.reg, lane.reg));
                    self.release(lane);
                }
                self.release(value);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            Intrinsic::Shuffle => {
                let value = self.expr(&args[0])?;
                self.protect(value);
                let lane = self.expr(&args[1])?;
                self.unprotect(value);
                let out = self.alloc_reg(span)?;
                self.instr(&format!("SHUFFLE r{out}, r{}, r{}", value.reg, lane.reg));
                self.release(value);
                self.release(lane);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            Intrinsic::ShuffleXor => {
                let value = self.expr(&args[0])?;
                let TypedExprKind::Literal(mask) = args[1].kind else {
                    return Err(Diagnostic::new(
                        span,
                        "internal error: non-constant shuffle mask",
                    ));
                };
                let out = self.alloc_reg(span)?;
                self.instr(&format!("SHUFFLE_XOR r{out}, r{}, {mask}", value.reg));
                self.release(value);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            Intrinsic::Ballot => {
                let predicate = self.expr(&args[0])?;
                let bit = self.alloc_reg(span)?;
                let selected = self.alloc_reg(span)?;
                self.instr(&format!("MOV r{bit}, 1"));
                self.instr(&format!("SHL r{bit}, r{bit}, r{LANE_REG}"));
                self.instr(&format!("CMP_NE p0, r{}, 0", predicate.reg));
                self.release(predicate);
                self.instr(&format!("MOV r{selected}, 0"));
                self.guarded_instr(0, false, &format!("MOV r{selected}, r{bit}"));
                self.free_reg(bit);
                let out = self.alloc_reg(span)?;
                self.instr(&format!("REDUCE_OR r{out}, r{selected}"));
                self.free_reg(selected);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            Intrinsic::Any | Intrinsic::All => {
                let value = self.expr(&args[0])?;
                let normalized = self.alloc_reg(span)?;
                self.instr(&format!("CMP_NE p0, r{}, 0", value.reg));
                self.release(value);
                self.instr(&format!("MOV r{normalized}, 0"));
                self.guarded_instr(0, false, &format!("MOV r{normalized}, 1"));
                let out = self.alloc_reg(span)?;
                let mnemonic = if intrinsic == Intrinsic::Any {
                    "REDUCE_OR"
                } else {
                    "REDUCE_AND"
                };
                self.instr(&format!("{mnemonic} r{out}, r{normalized}"));
                self.free_reg(normalized);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            Intrinsic::ReduceAdd
            | Intrinsic::ReduceAddUnsigned
            | Intrinsic::ReduceMinUnsigned
            | Intrinsic::ReduceMaxUnsigned
            | Intrinsic::ReduceAnd
            | Intrinsic::ReduceOr
            | Intrinsic::ReduceXor => {
                let value = self.expr(&args[0])?;
                let out = self.alloc_reg(span)?;
                let mnemonic = match intrinsic {
                    Intrinsic::ReduceAdd | Intrinsic::ReduceAddUnsigned => "REDUCE_ADD",
                    Intrinsic::ReduceMinUnsigned => "REDUCE_MIN",
                    Intrinsic::ReduceMaxUnsigned => "REDUCE_MAX",
                    Intrinsic::ReduceAnd => "REDUCE_AND",
                    Intrinsic::ReduceOr => "REDUCE_OR",
                    Intrinsic::ReduceXor => "REDUCE_XOR",
                    _ => unreachable!(),
                };
                self.instr(&format!("{mnemonic} r{out}, r{}", value.reg));
                self.release(value);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            Intrinsic::ReduceMinSigned | Intrinsic::ReduceMaxSigned => {
                let value = self.expr(&args[0])?;
                let biased = self.alloc_reg(span)?;
                self.instr(&format!("XOR r{biased}, r{}, {SIGN_BIT}", value.reg));
                self.release(value);
                let out = self.alloc_reg(span)?;
                let mnemonic = if intrinsic == Intrinsic::ReduceMinSigned {
                    "REDUCE_MIN"
                } else {
                    "REDUCE_MAX"
                };
                self.instr(&format!("{mnemonic} r{out}, r{biased}"));
                self.free_reg(biased);
                self.instr(&format!("XOR r{out}, r{out}, {SIGN_BIT}"));
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            Intrinsic::MinUnsigned
            | Intrinsic::MaxUnsigned
            | Intrinsic::MinSigned
            | Intrinsic::MaxSigned => {
                let left = self.expr(&args[0])?;
                self.protect(left);
                let right = self.expr(&args[1])?;
                self.unprotect(left);
                let signed = matches!(intrinsic, Intrinsic::MinSigned | Intrinsic::MaxSigned);
                let maximum = matches!(intrinsic, Intrinsic::MaxSigned | Intrinsic::MaxUnsigned);
                let (left, right) = if signed {
                    let biased_left = self.alloc_reg(span)?;
                    self.instr(&format!("XOR r{biased_left}, r{}, {SIGN_BIT}", left.reg));
                    self.release(left);
                    let biased_right = self.alloc_reg(span)?;
                    self.instr(&format!("XOR r{biased_right}, r{}, {SIGN_BIT}", right.reg));
                    self.release(right);
                    (
                        Value {
                            reg: biased_left,
                            owned: true,
                        },
                        Value {
                            reg: biased_right,
                            owned: true,
                        },
                    )
                } else {
                    (left, right)
                };
                let out = self.alloc_reg(span)?;
                let mnemonic = if maximum { "MAX" } else { "MIN" };
                self.instr(&format!("{mnemonic} r{out}, r{}, r{}", left.reg, right.reg));
                self.release(left);
                self.release(right);
                if signed {
                    self.instr(&format!("XOR r{out}, r{out}, {SIGN_BIT}"));
                }
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
        }
    }

    fn conditional(
        &mut self,
        condition: &TypedExpr,
        then_expr: &TypedExpr,
        else_expr: &TypedExpr,
        expr: &TypedExpr,
    ) -> Result<Value, Diagnostic> {
        if condition.uniformity == Uniformity::Divergent || self.current_guard.is_some() {
            return self.masked_conditional(condition, then_expr, else_expr, expr.span);
        }

        let otherwise = self.label("conditional_else");
        let end = self.label("conditional_end");
        self.jump_if_zero(condition, &otherwise)?;
        if expr.ty == Type::VOID {
            let then_value = self.expr(then_expr)?;
            self.release(then_value);
            self.instr(&format!("JMP {end}"));
            self.line(&format!("{otherwise}:"));
            let else_value = self.expr(else_expr)?;
            self.release(else_value);
            self.line(&format!("{end}:"));
            return Ok(Value {
                reg: 0,
                owned: false,
            });
        }
        let out = self.alloc_reg(expr.span)?;
        let then_value = self.expr(then_expr)?;
        self.instr(&format!("MOV r{out}, r{}", then_value.reg));
        self.release(then_value);
        self.instr(&format!("JMP {end}"));
        self.line(&format!("{otherwise}:"));
        let else_value = self.expr(else_expr)?;
        self.instr(&format!("MOV r{out}, r{}", else_value.reg));
        self.release(else_value);
        self.line(&format!("{end}:"));
        Ok(Value {
            reg: out,
            owned: true,
        })
    }

    fn masked_conditional(
        &mut self,
        condition: &TypedExpr,
        then_expr: &TypedExpr,
        else_expr: &TypedExpr,
        span: Span,
    ) -> Result<Value, Diagnostic> {
        let mask = match self.mask_stack.len() {
            0 => 3,
            1 => 2,
            _ => {
                return Err(Diagnostic::new(
                    span,
                    "divergent conditional nesting exhausted predicate masks",
                ))
            }
        };
        let parent = self.current_guard;
        let condition_value = self.expr(condition)?;
        if let Some(parent) = parent {
            self.raw_instr(&format!("BALLOT p0, r{}", condition_value.reg));
            self.raw_instr(&format!("ANDMASK p{mask}, p{parent}, p0"));
        } else {
            self.raw_instr(&format!("BALLOT p{mask}, r{}", condition_value.reg));
        }
        self.release(condition_value);

        let out = if then_expr.ty == Type::VOID {
            None
        } else {
            let out = self.alloc_reg(span)?;
            self.instr(&format!("MOV r{out}, 0"));
            Some(out)
        };
        self.mask_stack.push(mask);
        self.current_guard = Some(mask);
        let then_value = self.expr(then_expr)?;
        if let Some(out) = out {
            self.instr(&format!("MOV r{out}, r{}", then_value.reg));
        }
        self.release(then_value);

        if let Some(parent) = parent {
            self.raw_instr(&format!("NOTMASK p0, p{mask}"));
            self.raw_instr(&format!("ANDMASK p{mask}, p{parent}, p0"));
        } else {
            self.raw_instr(&format!("NOTMASK p{mask}, p{mask}"));
        }
        let else_value = self.expr(else_expr)?;
        if let Some(out) = out {
            self.instr(&format!("MOV r{out}, r{}", else_value.reg));
        }
        self.release(else_value);

        self.mask_stack.pop();
        self.current_guard = parent;
        Ok(out.map_or(
            Value {
                reg: 0,
                owned: false,
            },
            |reg| Value { reg, owned: true },
        ))
    }

    fn call(
        &mut self,
        function: FunctionId,
        args: &[TypedExpr],
        return_type: Type,
        span: Span,
        call_site: usize,
    ) -> Result<Value, Diagnostic> {
        let values = self.expr_values(args)?;

        let semantic_live = self.call_live_out.get(&call_site).cloned().ok_or_else(|| {
            Diagnostic::new(
                span,
                "internal error: missing call-site live-out information",
            )
        })?;
        let mut semantic_locals: Vec<LocalId> = semantic_live.iter().copied().collect();
        semantic_locals.sort_unstable();
        let mut saved = Vec::new();
        let mut saved_scalars = Vec::new();
        let mut saved_vector_homes = Vec::new();
        let mut saved_scalar_homes = Vec::new();
        let mut memory_resident = Vec::new();
        for &local in &semantic_locals {
            match self.local_storage[local] {
                Some(LocalStorage::Vector(reg)) => {
                    if !saved.contains(&reg) {
                        saved.push(reg);
                    }
                    saved_vector_homes.push(format!("{}->r{reg}", self.locals[local].name));
                }
                Some(LocalStorage::Scalar(reg)) => {
                    if !saved_scalars.contains(&reg) {
                        saved_scalars.push(reg);
                    }
                    saved_scalar_homes.push(format!("{}->s{reg}", self.locals[local].name));
                }
                Some(LocalStorage::Frame(offset)) => {
                    memory_resident.push(format!("{}->frame[{offset}]", self.locals[local].name));
                }
                None => {}
            }
        }
        let protected_vectors: Vec<u8> = self
            .protected
            .iter()
            .enumerate()
            .filter_map(|(reg, count)| (*count != 0).then_some(reg as u8))
            .collect();
        for &reg in &protected_vectors {
            if !saved.contains(&reg) {
                saved.push(reg);
            }
        }
        saved.sort_unstable();
        saved_scalars.sort_unstable();
        // Stage every argument in ABI-owned stack slots. The old lowering
        // reloaded arguments from their incidental caller-save slots, which
        // was invalid for values such as WARP in reserved r13 and needlessly
        // coupled expression lowering to the caller-save register set.
        let argument_base = saved.len() + saved_scalars.len();
        let return_slot = (return_type != Type::VOID).then_some(argument_base + values.len());
        let frame_slots = argument_base + values.len() + usize::from(return_slot.is_some());
        let frame_words = frame_slots * 32;
        let allocator_active_vector_homes: Vec<String> = self
            .local_active
            .iter()
            .enumerate()
            .filter_map(
                |(local, active)| match (*active, self.local_storage[local]) {
                    (true, Some(LocalStorage::Vector(reg))) => {
                        Some(format!("r{reg}={}", self.locals[local].name))
                    }
                    _ => None,
                },
            )
            .collect();
        let allocator_active_scalar_homes: Vec<String> = self
            .local_active
            .iter()
            .enumerate()
            .filter_map(
                |(local, active)| match (*active, self.local_storage[local]) {
                    (true, Some(LocalStorage::Scalar(reg))) => {
                        Some(format!("s{reg}={}", self.locals[local].name))
                    }
                    _ => None,
                },
            )
            .collect();
        let semantic_names: Vec<&str> = semantic_locals
            .iter()
            .map(|local| self.locals[*local].name.as_str())
            .collect();
        let protected_names: Vec<String> = protected_vectors
            .iter()
            .map(|reg| format!("r{reg}"))
            .collect();
        let target_name = self.function_names[function].clone();
        self.line(&format!(
            "; call {target_name}: allocator_active_vector_homes={} [{}] allocator_active_scalar_homes={} [{}]",
            allocator_active_vector_homes.len(),
            allocator_active_vector_homes.join(", "),
            allocator_active_scalar_homes.len(),
            allocator_active_scalar_homes.join(", ")
        ));
        self.line(&format!(
            "; call {target_name}: semantic_live_out={} [{}] saved_vector_homes={} [{}] saved_scalar_homes={} [{}] memory_resident={} [{}] protected_temporaries={} [{}]",
            semantic_names.len(),
            semantic_names.join(", "),
            saved_vector_homes.len(),
            saved_vector_homes.join(", "),
            saved_scalar_homes.len(),
            saved_scalar_homes.join(", "),
            memory_resident.len(),
            memory_resident.join(", "),
            protected_names.len(),
            protected_names.join(", ")
        ));
        self.line(&format!(
            "; call {target_name}: saved_vector_slots={} saved_scalar_slots={} argument_slots={} return_slots={} call_frame_slots={} call_frame_words={}",
            saved.len(),
            saved_scalars.len(),
            values.len(),
            usize::from(return_slot.is_some()),
            frame_slots,
            frame_words
        ));
        self.line(&format!(
            "; call {target_name}: dynamic_memory_instructions={} dynamic_lane_word_accesses={} caller_fixed_frame_words_per_lane={} allocator_spill_slots_per_lane={}",
            frame_slots * 2,
            frame_slots * 64,
            self.current_frame_words,
            self.allocation_spills
        ));
        if frame_words != 0 {
            self.instr(&format!(
                "S_ADD_I {STACK_POINTER}, {STACK_POINTER}, -{frame_words}"
            ));
        }
        for (slot, &reg) in saved.iter().enumerate() {
            self.stack_address(slot);
            self.instr(&format!("STORE r{STACK_ADDR_REG}, r{reg}"));
        }
        for (index, &reg) in saved_scalars.iter().enumerate() {
            self.stack_address(saved.len() + index);
            self.instr(&format!("S_BCAST r15, s{reg}"));
            self.instr(&format!("STORE r{STACK_ADDR_REG}, r15"));
        }
        for (index, value) in values.iter().enumerate() {
            self.stack_address(argument_base + index);
            self.instr(&format!("STORE r{STACK_ADDR_REG}, r{}", value.reg));
        }
        for arg_reg in 0..values.len() {
            self.stack_address(argument_base + arg_reg);
            self.instr(&format!("LOAD r{arg_reg}, r{STACK_ADDR_REG}"));
        }
        let target = self.function_labels[function].clone();
        self.instr(&format!("CALL {target}"));
        if let Some(slot) = return_slot {
            self.stack_address(slot);
            self.instr(&format!("STORE r{STACK_ADDR_REG}, r0"));
        }
        for (slot, &reg) in saved.iter().enumerate() {
            self.stack_address(slot);
            self.instr(&format!("LOAD r{reg}, r{STACK_ADDR_REG}"));
        }
        for (index, &reg) in saved_scalars.iter().enumerate() {
            self.stack_address(saved.len() + index);
            self.instr(&format!("LOAD r15, r{STACK_ADDR_REG}"));
            self.instr(&format!("S_GET s{reg}, r15"));
        }
        for value in values {
            self.release(value);
        }

        let result = if let Some(slot) = return_slot {
            let out = self.alloc_reg(span)?;
            self.stack_address(slot);
            self.instr(&format!("LOAD r{out}, r{STACK_ADDR_REG}"));
            Value {
                reg: out,
                owned: true,
            }
        } else {
            Value {
                reg: 0,
                owned: false,
            }
        };
        if frame_words != 0 {
            self.instr(&format!(
                "S_ADD_I {STACK_POINTER}, {STACK_POINTER}, {frame_words}"
            ));
        }
        Ok(result)
    }

    fn stack_address(&mut self, slot: usize) {
        self.instr(&format!("S_BCAST r{STACK_ADDR_REG}, {STACK_POINTER}"));
        let offset = slot * 32;
        if offset != 0 {
            self.instr(&format!(
                "ADD r{STACK_ADDR_REG}, r{STACK_ADDR_REG}, {offset}"
            ));
        }
        self.instr(&format!(
            "ADD r{STACK_ADDR_REG}, r{STACK_ADDR_REG}, r{LANE_REG}"
        ));
    }

    fn unary(
        &mut self,
        op: UnaryOp,
        operand: &TypedExpr,
        expr: &TypedExpr,
    ) -> Result<Value, Diagnostic> {
        let value = self.expr(operand)?;
        match op {
            UnaryOp::Plus => Ok(value),
            UnaryOp::Minus | UnaryOp::BitNot => {
                let out = self.alloc_reg(expr.span)?;
                let mnemonic = if op == UnaryOp::Minus { "NEG" } else { "NOT" };
                self.instr(&format!("{mnemonic} r{out}, r{}", value.reg));
                self.release(value);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            UnaryOp::LogicalNot => {
                let out = self.materialize_compare("CMP_EQ", value.reg, None, 0, expr.span)?;
                self.release(value);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            _ => unreachable!("increment/decrement is represented separately by semantic analysis"),
        }
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        left_expr: &TypedExpr,
        right_expr: &TypedExpr,
        operand_type: Type,
        expr: &TypedExpr,
    ) -> Result<Value, Diagnostic> {
        if op == BinaryOp::Comma {
            let left = self.expr(left_expr)?;
            self.release(left);
            return self.expr(right_expr);
        }
        if matches!(op, BinaryOp::LogicalAnd | BinaryOp::LogicalOr) {
            return self.logical(op, left_expr, right_expr, expr.span);
        }
        let left = self.expr(left_expr)?;
        self.protect(left);
        let right = self.expr(right_expr)?;
        self.unprotect(left);
        let out = self.emit_binary(op, left.reg, right.reg, operand_type, expr.span)?;
        self.release(left);
        self.release(right);
        Ok(Value {
            reg: out,
            owned: true,
        })
    }

    fn logical(
        &mut self,
        op: BinaryOp,
        left_expr: &TypedExpr,
        right_expr: &TypedExpr,
        span: Span,
    ) -> Result<Value, Diagnostic> {
        let out = self.alloc_reg(span)?;
        let end = self.label("logical_end");
        let left = self.expr(left_expr)?;
        self.instr(&format!("CMP_NE p0, r{}, 0", left.reg));
        self.release(left);
        match op {
            BinaryOp::LogicalAnd => {
                self.instr(&format!("MOV r{out}, 0"));
                self.instr("NOTMASK p1, p0");
                self.instr(&format!("JMP_IF_ANY p1, {end}"));
                self.protect(Value {
                    reg: out,
                    owned: true,
                });
                let right = self.expr(right_expr)?;
                self.unprotect(Value {
                    reg: out,
                    owned: true,
                });
                self.instr(&format!("CMP_NE p0, r{}, 0", right.reg));
                self.release(right);
                self.guarded_instr(0, false, &format!("MOV r{out}, 1"));
            }
            BinaryOp::LogicalOr => {
                self.instr(&format!("MOV r{out}, 1"));
                self.instr(&format!("JMP_IF_ANY p0, {end}"));
                self.protect(Value {
                    reg: out,
                    owned: true,
                });
                let right = self.expr(right_expr)?;
                self.unprotect(Value {
                    reg: out,
                    owned: true,
                });
                self.instr(&format!("CMP_EQ p0, r{}, 0", right.reg));
                self.release(right);
                self.guarded_instr(0, false, &format!("MOV r{out}, 0"));
            }
            _ => unreachable!(),
        }
        self.line(&format!("{end}:"));
        Ok(Value {
            reg: out,
            owned: true,
        })
    }

    fn assignment(
        &mut self,
        target: &LValue,
        op: AssignOp,
        right_expr: &TypedExpr,
        operation_type: Type,
        scale: usize,
        expr: &TypedExpr,
    ) -> Result<Value, Diagnostic> {
        let right = self.expr(right_expr)?;
        self.protect(right);
        if op == AssignOp::Assign {
            self.store_lvalue(target, right.reg, expr.span)?;
            self.unprotect(right);
            Ok(right)
        } else {
            let current = self.load_lvalue(target, expr.span)?;
            self.unprotect(right);
            let right = if scale == 1 {
                right
            } else {
                let scaled = self.alloc_reg(expr.span)?;
                self.instr(&format!("MUL r{scaled}, r{}, {scale}", right.reg));
                self.release(right);
                Value {
                    reg: scaled,
                    owned: true,
                }
            };
            let binary = assignment_binary(op);
            let out =
                self.emit_binary(binary, current.reg, right.reg, operation_type, expr.span)?;
            self.release(current);
            self.release(right);
            let output = Value {
                reg: out,
                owned: true,
            };
            self.protect(output);
            self.store_lvalue(target, out, expr.span)?;
            self.unprotect(output);
            Ok(output)
        }
    }

    fn emit_binary(
        &mut self,
        op: BinaryOp,
        left: u8,
        right: u8,
        operand_type: Type,
        span: Span,
    ) -> Result<u8, Diagnostic> {
        use BinaryOp::*;
        match op {
            Add | Sub | Mul | BitAnd | BitOr | BitXor | Shl => {
                let out = self.alloc_reg(span)?;
                let mnemonic = match op {
                    Add => "ADD",
                    Sub => "SUB",
                    Mul => "MUL",
                    BitAnd => "AND",
                    BitOr => "OR",
                    BitXor => "XOR",
                    Shl => "SHL",
                    _ => unreachable!(),
                };
                self.instr(&format!("{mnemonic} r{out}, r{left}, r{right}"));
                Ok(out)
            }
            Div | Mod if operand_type == Type::I32 => self.signed_div_mod(op, left, right, span),
            Div | Mod => {
                let out = self.alloc_reg(span)?;
                let mnemonic = if op == Div { "DIV" } else { "MOD" };
                self.instr(&format!("{mnemonic} r{out}, r{left}, r{right}"));
                Ok(out)
            }
            Shr if operand_type == Type::I32 => self.arithmetic_shift_right(left, right, span),
            Shr => {
                let out = self.alloc_reg(span)?;
                self.instr(&format!("SHR r{out}, r{left}, r{right}"));
                Ok(out)
            }
            Eq | Ne => {
                let mnemonic = if op == Eq { "CMP_EQ" } else { "CMP_NE" };
                self.materialize_compare(mnemonic, left, Some(right), 0, span)
            }
            Lt | Le | Gt | Ge => self.relational(op, left, right, operand_type, span),
            LogicalAnd | LogicalOr | Comma => unreachable!(),
        }
    }

    fn signed_div_mod(
        &mut self,
        op: BinaryOp,
        left: u8,
        right: u8,
        span: Span,
    ) -> Result<u8, Diagnostic> {
        let out = self.alloc_reg(span)?;
        let temp = self.alloc_reg(span)?;
        self.instr(&format!("ABS r{out}, r{left}"));
        self.instr(&format!("ABS r{temp}, r{right}"));
        let mnemonic = if op == BinaryOp::Div { "DIV" } else { "MOD" };
        self.instr(&format!("{mnemonic} r{out}, r{out}, r{temp}"));
        if op == BinaryOp::Div {
            self.instr(&format!("XOR r{temp}, r{left}, r{right}"));
        } else {
            self.instr(&format!("MOV r{temp}, r{left}"));
        }
        self.instr(&format!("CMP_GE p0, r{temp}, {SIGN_BIT}"));
        self.guarded_instr(0, false, &format!("NEG r{out}, r{out}"));
        self.free_reg(temp);
        Ok(out)
    }

    fn arithmetic_shift_right(
        &mut self,
        left: u8,
        right: u8,
        span: Span,
    ) -> Result<u8, Diagnostic> {
        let out = self.alloc_reg(span)?;
        let fill = self.alloc_reg(span)?;
        let neg_shift = self.alloc_reg(span)?;
        self.instr(&format!("SHR r{out}, r{left}, r{right}"));
        self.instr(&format!("SHR r{fill}, r{left}, 31"));
        self.instr(&format!("NEG r{fill}, r{fill}"));
        self.instr(&format!("NEG r{neg_shift}, r{right}"));
        self.instr(&format!("SHL r{fill}, r{fill}, r{neg_shift}"));
        self.instr(&format!("CMP_EQ p0, r{right}, 0"));
        self.guarded_instr(0, true, &format!("OR r{out}, r{out}, r{fill}"));
        self.free_reg(neg_shift);
        self.free_reg(fill);
        Ok(out)
    }

    fn relational(
        &mut self,
        op: BinaryOp,
        left: u8,
        right: u8,
        ty: Type,
        span: Span,
    ) -> Result<u8, Diagnostic> {
        let mnemonic = match op {
            BinaryOp::Lt => "CMP_LT",
            BinaryOp::Le => "CMP_LE",
            BinaryOp::Gt => "CMP_GT",
            BinaryOp::Ge => "CMP_GE",
            _ => unreachable!(),
        };
        if ty != Type::I32 {
            return self.materialize_compare(mnemonic, left, Some(right), 0, span);
        }
        let biased_left = self.alloc_reg(span)?;
        let biased_right = self.alloc_reg(span)?;
        self.instr(&format!("XOR r{biased_left}, r{left}, {SIGN_BIT}"));
        self.instr(&format!("XOR r{biased_right}, r{right}, {SIGN_BIT}"));
        let out = self.materialize_compare(mnemonic, biased_left, Some(biased_right), 0, span)?;
        self.free_reg(biased_right);
        self.free_reg(biased_left);
        Ok(out)
    }

    fn materialize_compare(
        &mut self,
        mnemonic: &str,
        left: u8,
        right: Option<u8>,
        immediate: u32,
        span: Span,
    ) -> Result<u8, Diagnostic> {
        match right {
            Some(right) => self.instr(&format!("{mnemonic} p0, r{left}, r{right}")),
            None => self.instr(&format!("{mnemonic} p0, r{left}, {immediate}")),
        }
        let out = self.alloc_reg(span)?;
        self.instr(&format!("MOV r{out}, 0"));
        self.guarded_instr(0, false, &format!("MOV r{out}, 1"));
        Ok(out)
    }

    fn emit_return(&mut self) {
        if self.current_frame_words != 0 {
            self.instr(&format!(
                "S_ADD_I {STACK_POINTER}, {STACK_POINTER}, {}",
                self.current_frame_words * 32
            ));
        }
        if self.current_function == Some(self.main_function) {
            self.instr("HALT");
        } else {
            self.instr("RET");
        }
    }

    fn local_address(&mut self, local: LocalId, span: Span) -> Result<Value, Diagnostic> {
        self.local_address_offset(local, 0, span)
    }

    fn local_address_offset(
        &mut self,
        local: LocalId,
        extra: usize,
        span: Span,
    ) -> Result<Value, Diagnostic> {
        let offset = match self.local_storage[local] {
            Some(LocalStorage::Frame(offset)) => offset + extra,
            _ => {
                return Err(Diagnostic::new(
                    span,
                    "cannot take address of register-only local",
                ))
            }
        };
        let out = self.alloc_reg(span)?;
        self.instr(&format!("S_BCAST r{out}, {STACK_POINTER}"));
        if self.current_frame_words != 0 {
            self.instr(&format!(
                "MUL r{STACK_ADDR_REG}, r{LANE_REG}, {}",
                self.current_frame_words
            ));
            self.instr(&format!("ADD r{out}, r{out}, r{STACK_ADDR_REG}"));
        }
        if offset != 0 {
            self.instr(&format!("ADD r{out}, r{out}, {offset}"));
        }
        Ok(Value {
            reg: out,
            owned: true,
        })
    }

    fn lvalue_address(&mut self, lvalue: &LValue, span: Span) -> Result<Value, Diagnostic> {
        match lvalue {
            LValue::Local(local) => self.local_address(*local, span),
            LValue::Global(global) => {
                let out = self.alloc_reg(span)?;
                self.instr(&format!("MOV r{out}, {}", self.globals[*global].address));
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            LValue::Deref(pointer) => self.expr(pointer),
            LValue::Index { base, index, scale } => {
                let base = self.expr(base)?;
                self.protect(base);
                let index = self.expr(index)?;
                self.unprotect(base);
                let scaled = if *scale == 1 {
                    index
                } else {
                    let out = self.alloc_reg(span)?;
                    self.instr(&format!("MUL r{out}, r{}, {scale}", index.reg));
                    self.release(index);
                    Value {
                        reg: out,
                        owned: true,
                    }
                };
                let out = self.alloc_reg(span)?;
                self.instr(&format!("ADD r{out}, r{}, r{}", base.reg, scaled.reg));
                self.release(base);
                self.release(scaled);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
            LValue::Member { base, offset } => {
                let address = self.lvalue_address(base, span)?;
                if *offset == 0 {
                    return Ok(address);
                }
                let out = self.alloc_reg(span)?;
                self.instr(&format!("ADD r{out}, r{}, {offset}", address.reg));
                self.release(address);
                Ok(Value {
                    reg: out,
                    owned: true,
                })
            }
        }
    }

    fn load_lvalue(&mut self, lvalue: &LValue, span: Span) -> Result<Value, Diagnostic> {
        if let LValue::Local(local) = lvalue {
            match self.local_storage[*local] {
                Some(LocalStorage::Vector(reg)) => {
                    return Ok(Value { reg, owned: false });
                }
                Some(LocalStorage::Scalar(reg)) => {
                    let out = self.alloc_reg(span)?;
                    self.instr(&format!("S_BCAST r{out}, s{reg}"));
                    return Ok(Value {
                        reg: out,
                        owned: true,
                    });
                }
                Some(LocalStorage::Frame(_)) => {}
                None => {
                    return Err(Diagnostic::new(
                        span,
                        "internal error: local has no allocated storage",
                    ));
                }
            }
        }
        let address = self.lvalue_address(lvalue, span)?;
        let out = self.alloc_reg(span)?;
        self.instr(&format!("LOAD r{out}, r{}", address.reg));
        self.release(address);
        Ok(Value {
            reg: out,
            owned: true,
        })
    }

    fn store_lvalue(&mut self, lvalue: &LValue, source: u8, span: Span) -> Result<(), Diagnostic> {
        if let LValue::Local(local) = lvalue {
            if !matches!(self.local_storage[*local], Some(LocalStorage::Frame(_))) {
                return self.store_local(*local, source, span);
            }
        }
        let address = self.lvalue_address(lvalue, span)?;
        self.instr(&format!("STORE r{}, r{source}", address.reg));
        self.release(address);
        Ok(())
    }

    fn store_local(&mut self, local: LocalId, source: u8, span: Span) -> Result<(), Diagnostic> {
        match self.local_storage[local] {
            Some(LocalStorage::Vector(target)) => {
                if target != source {
                    self.instr(&format!("MOV r{target}, r{source}"));
                }
            }
            Some(LocalStorage::Scalar(target)) => {
                if self.current_guard.is_some() {
                    return Err(Diagnostic::new(
                        span,
                        "internal error: divergent value allocated to scalar register",
                    ));
                }
                self.instr(&format!("S_GET s{target}, r{source}"));
            }
            Some(LocalStorage::Frame(_)) => {
                let address = self.local_address(local, span)?;
                self.instr(&format!("STORE r{}, r{source}", address.reg));
                self.release(address);
            }
            None => {
                return Err(Diagnostic::new(
                    span,
                    "internal error: local has no allocated storage",
                ));
            }
        }
        Ok(())
    }

    fn alloc_reg(&mut self, span: Span) -> Result<u8, Diagnostic> {
        let Some(reg) = self.used.iter().position(|used| !*used) else {
            return Err(Diagnostic::new(
                span,
                "temporary vector-register demand exceeds r0-r12 after liveness allocation and spilling",
            ));
        };
        self.used[reg] = true;
        self.update_peak_vector_regs();
        Ok(reg as u8)
    }

    fn free_reg(&mut self, reg: u8) {
        debug_assert!((reg as usize) < ALLOCATABLE_REGS && self.used[reg as usize]);
        self.used[reg as usize] = false;
    }

    fn release(&mut self, value: Value) {
        if value.owned {
            self.free_reg(value.reg);
        }
    }

    fn protect(&mut self, value: Value) {
        if (value.reg as usize) < ALLOCATABLE_REGS {
            self.protected[value.reg as usize] += 1;
        }
    }

    fn unprotect(&mut self, value: Value) {
        if (value.reg as usize) < ALLOCATABLE_REGS {
            debug_assert!(self.protected[value.reg as usize] != 0);
            self.protected[value.reg as usize] -= 1;
        }
    }

    fn expr_values(&mut self, expressions: &[TypedExpr]) -> Result<Vec<Value>, Diagnostic> {
        let mut values = Vec::with_capacity(expressions.len());
        for expression in expressions {
            let value = self.expr(expression)?;
            self.protect(value);
            values.push(value);
        }
        for &value in &values {
            self.unprotect(value);
        }
        Ok(values)
    }

    fn update_peak_vector_regs(&mut self) {
        self.peak_vector_regs = self
            .peak_vector_regs
            .max(self.used.iter().filter(|used| **used).count());
    }

    fn advance_allocation(&mut self, point: usize) -> Result<(), Diagnostic> {
        for local in 0..self.local_active.len() {
            if !self.local_active[local] || self.local_ends[local].unwrap_or(0) >= point {
                continue;
            }
            match self.local_storage[local] {
                Some(LocalStorage::Vector(reg)) => self.free_reg(reg),
                Some(LocalStorage::Scalar(reg)) => self.scalar_used[reg as usize] = false,
                _ => {}
            }
            self.local_active[local] = false;
        }
        for local in 0..self.local_active.len() {
            if self.local_active[local]
                || self.local_starts[local].is_none()
                || self.local_starts[local].unwrap() > point
                || self.local_ends[local].unwrap() < point
            {
                continue;
            }
            match self.local_storage[local] {
                Some(LocalStorage::Vector(reg)) => {
                    if self.used[reg as usize] {
                        return Err(Diagnostic::new(
                            self.locals[local].span,
                            "internal error: overlapping vector-register allocation",
                        ));
                    }
                    self.used[reg as usize] = true;
                }
                Some(LocalStorage::Scalar(reg)) => {
                    if self.scalar_used[reg as usize] {
                        return Err(Diagnostic::new(
                            self.locals[local].span,
                            "internal error: overlapping scalar-register allocation",
                        ));
                    }
                    self.scalar_used[reg as usize] = true;
                }
                _ => {}
            }
            self.local_active[local] = true;
        }
        self.update_peak_vector_regs();
        Ok(())
    }

    fn label(&mut self, stem: &str) -> String {
        let label = format!("__warpc_{stem}_{}", self.next_label);
        self.next_label += 1;
        label
    }

    fn line(&mut self, text: &str) {
        self.assembly.push_str(text);
        self.assembly.push('\n');
    }

    fn instr(&mut self, text: &str) {
        if let Some(predicate) = self.current_guard {
            self.raw_instr(&format!("@p{predicate} {text}"));
        } else {
            self.raw_instr(text);
        }
    }

    fn guarded_instr(&mut self, predicate: u8, inverted: bool, text: &str) {
        if let Some(execution) = self.current_guard {
            if inverted {
                self.raw_instr(&format!("NOTMASK p1, p{predicate}"));
                self.raw_instr(&format!("ANDMASK p1, p{execution}, p1"));
            } else {
                self.raw_instr(&format!("ANDMASK p1, p{execution}, p{predicate}"));
            }
            self.raw_instr(&format!("@p1 {text}"));
        } else {
            let bang = if inverted { "!" } else { "" };
            self.raw_instr(&format!("@{bang}p{predicate} {text}"));
        }
    }

    fn raw_instr(&mut self, text: &str) {
        self.assembly.push_str("    ");
        self.line(text);
    }
}

fn statement_span(statement: &TypedStmt) -> Span {
    match statement {
        TypedStmt::Decl { span, .. }
        | TypedStmt::If { span, .. }
        | TypedStmt::While { span, .. }
        | TypedStmt::DoWhile { span, .. }
        | TypedStmt::For { span, .. }
        | TypedStmt::Switch { span, .. }
        | TypedStmt::Case { span, .. }
        | TypedStmt::Default { span, .. }
        | TypedStmt::Return(_, span)
        | TypedStmt::Break(span)
        | TypedStmt::Continue(span) => *span,
        TypedStmt::Block(block) => block.span,
        TypedStmt::Expr(Some(expr)) => expr.span,
        TypedStmt::Expr(None) => Span::new(0, 1, 1),
    }
}

fn assignment_binary(op: AssignOp) -> BinaryOp {
    match op {
        AssignOp::Add => BinaryOp::Add,
        AssignOp::Sub => BinaryOp::Sub,
        AssignOp::Mul => BinaryOp::Mul,
        AssignOp::Div => BinaryOp::Div,
        AssignOp::Mod => BinaryOp::Mod,
        AssignOp::Shl => BinaryOp::Shl,
        AssignOp::Shr => BinaryOp::Shr,
        AssignOp::BitAnd => BinaryOp::BitAnd,
        AssignOp::BitXor => BinaryOp::BitXor,
        AssignOp::BitOr => BinaryOp::BitOr,
        AssignOp::Assign => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer, parser, sema};

    fn assembly(source: &str) -> String {
        let tokens = lexer::lex(source).unwrap();
        let ast = parser::parse(&tokens).unwrap();
        let typed = sema::analyze(ast).unwrap();
        generate(&typed).unwrap()
    }

    #[test]
    fn signed_operations_use_explicit_lowering() {
        let text = assembly(
            "int main(void) { int a = -21; int b = 4; return a / b + (a >> b) + (a < b); }",
        );
        assert!(text.contains("ABS"));
        assert!(text.contains("@p0 NEG"));
        assert!(text.contains("@!p0 OR"));
        assert!(text.contains("2147483648"));
    }

    #[test]
    fn logical_and_is_short_circuit_control_flow() {
        let text = assembly("int main(void) { int x = 0; return x && ++x; }");
        assert!(text.contains("JMP_IF_ANY"));
        assert!(text.contains("logical_end"));
    }

    #[test]
    fn structured_control_emits_explicit_targets() {
        let text = assembly(
            "int main(void) { int x = 0; while (x < 3) { ++x; if (x == 2) continue; } do { --x; } while (x); for (int i = 0; i < 2; ++i) { if (i) continue; x += i; } return x; }",
        );
        assert!(text.contains("while_condition"));
        assert!(text.contains("do_condition"));
        assert!(text.contains("for_step"));
        assert!(text.contains("JMP __warpc_for_step"));
    }

    #[test]
    fn switch_emits_dispatch_and_case_labels() {
        let text = assembly(
            "int main(void) { int x = 2; switch (x) { case 1: x = 3; case 1 + 1: x = 42; break; default: x = 0; } return x; }",
        );
        assert!(text.contains("CMP_EQ p0"));
        assert!(text.contains("switch_case"));
        assert!(text.contains("switch_end"));
    }

    #[test]
    fn calls_use_the_lane_private_stack_abi() {
        let text = assembly(
            "int add(int, int); int main(void) { int keep = 20; return add(keep, 22); } int add(int a, int b) { if (a == 20) return a + b; return 0; }",
        );
        assert!(text.contains("LANEID r13"));
        assert!(text.contains("S_MOV_I s7, 16384"));
        assert!(text.contains("S_ADD_I s7, s7, -"));
        assert!(text.contains("STORE r14"));
        assert!(text.contains("CALL __warpc_fn_add"));
        assert!(text.contains("__warpc_fn_add:"));
        assert!(text.contains("RET"));
    }

    #[test]
    fn call_arguments_are_staged_independently_of_source_registers() {
        let text = assembly(
            "int f(int a,int b) { if(a<0) return b; return a+b; } int main(void) { int side=0; int local=WARP+4; int a=f(WARP,1); int b=f(WARP+1,++side); int c=f(WARP*3+7,WARP); int d=f(local,warp_lane_id()); return 42+a-a+b-b+c-c+d-d+side-side; }",
        );
        assert!(text.contains("CALL __warpc_fn_f"));
        // WARP and warp_lane_id both reside in reserved r13. They are now
        // valid STORE sources for dedicated argument staging slots.
        assert!(text.contains("STORE r14, r13"), "{text}");
        assert!(!text.contains("argument register was not saved"));
    }

    #[test]
    fn call_diagnostics_separate_save_transport_and_fixed_frames() {
        let almost_empty = assembly(
            "int id(int x) { if (x) return x; return 0; } int main(void) { return id(WARP); }",
        );
        assert!(almost_empty.contains(
            "; call id: saved_vector_slots=0 saved_scalar_slots=0 argument_slots=1 return_slots=1 call_frame_slots=2 call_frame_words=64"
        ), "{almost_empty}");
        assert!(almost_empty.contains(
            "; call id: dynamic_memory_instructions=4 dynamic_lane_word_accesses=128 caller_fixed_frame_words_per_lane=0 allocator_spill_slots_per_lane=0"
        ), "{almost_empty}");

        let live_caller = assembly(
            "int id(int x) { if (x) return x; return 0; } int main(void) { int a=WARP+1; int b=WARP+2; int c=WARP+3; int d=WARP+4; return id(WARP)+a+b+c+d; }",
        );
        assert!(live_caller.contains(
            "; call id: saved_vector_slots=4 saved_scalar_slots=0 argument_slots=1 return_slots=1 call_frame_slots=6 call_frame_words=192"
        ), "{live_caller}");
        assert!(
            live_caller.contains("allocator_active_vector_homes=4 [r0=a, r1=b, r2=c, r3=d]"),
            "{live_caller}"
        );

        let many_live = assembly(
            "int id(int x) { if (x) return x; return 0; } int main(void) { int a=WARP+1; int b=WARP+2; int c=WARP+3; int d=WARP+4; int e=WARP+5; int f=WARP+6; int u=11; int v=13; int w=17; return id(WARP)+a+b+c+d+e+f+u+v+w; }",
        );
        assert!(many_live.contains(
            "; call id: saved_vector_slots=6 saved_scalar_slots=3 argument_slots=1 return_slots=1 call_frame_slots=11 call_frame_words=352"
        ), "{many_live}");
        assert!(
            many_live.contains("allocator_active_scalar_homes=3 [s0=u, s1=v, s2=w]"),
            "{many_live}"
        );
    }

    #[test]
    fn call_live_out_is_definition_sensitive_across_control_flow() {
        let helper = "int id(int x) { if (x < 0) return 0; return x; }";

        let dead = assembly(&format!(
            "{helper} int main(void) {{ int a=WARP+1; id(3); a=7; return a; }}"
        ));
        assert!(dead.contains("; call id: semantic_live_out=0 []"), "{dead}");
        assert!(
            dead.contains("saved_vector_slots=0 saved_scalar_slots=0"),
            "{dead}"
        );

        for source in [
            format!("{helper} int main(void) {{ int a=WARP+1; id(3); return a; }}"),
            format!("{helper} int main(void) {{ int a=WARP+1; id(3); a=a+1; return a; }}"),
        ] {
            let live = assembly(&source);
            assert!(live.contains("semantic_live_out=1 [a]"), "{live}");
            assert!(live.contains("saved_vector_homes=1 [a->r0]"), "{live}");
        }

        let branch_union = assembly(&format!(
            "{helper} int main(void) {{ int a=WARP+1; int b=WARP+2; int out; id(3); if(warp_vm_id()) out=a; else out=b; return out; }}"
        ));
        assert!(
            branch_union.contains("semantic_live_out=2 [a, b]"),
            "{branch_union}"
        );

        let branch_kill = assembly(&format!(
            "{helper} int main(void) {{ int a=WARP+1; id(3); if(warp_vm_id()) a=1; else a=2; return a; }}"
        ));
        assert!(
            branch_kill.contains("; call id: semantic_live_out=0 []"),
            "{branch_kill}"
        );

        let loop_carried = assembly(&format!(
            "{helper} int main(void) {{ int a=WARP; for(int i=0;i<3;++i) a=a+id(i); return a; }}"
        ));
        assert!(
            loop_carried.contains("semantic_live_out=2 [a, i]"),
            "{loop_carried}"
        );
        assert!(
            loop_carried.contains("saved_vector_homes=1 [a->r0] saved_scalar_homes=1 [i->s0]"),
            "{loop_carried}"
        );

        let while_carried = assembly(&format!(
            "{helper} int main(void) {{ int a=WARP; int i=0; while(i<3) {{ a=a+id(i); ++i; }} return a; }}"
        ));
        assert!(
            while_carried.contains("semantic_live_out=2 [a, i]"),
            "{while_carried}"
        );

        let do_carried = assembly(&format!(
            "{helper} int main(void) {{ int a=WARP; int i=0; do {{ a=a+id(i); ++i; }} while(i<3); return a; }}"
        ));
        assert!(
            do_carried.contains("semantic_live_out=2 [a, i]"),
            "{do_carried}"
        );

        let switch_union = assembly(&format!(
            "{helper} int main(void) {{ int a=WARP+1; int b=WARP+2; int out; id(3); switch(warp_vm_id()) {{ case 0: out=a; break; default: out=b; }} return out; }}"
        ));
        assert!(
            switch_union.contains("semantic_live_out=2 [a, b]"),
            "{switch_union}"
        );

        let many_dead = assembly(&format!(
            "{helper} int main(void) {{ int a=WARP+1; int b=WARP+2; int c=WARP+3; int d=WARP+4; int e=WARP+5; int f=WARP+6; id(3); return 42; }}"
        ));
        assert!(
            many_dead.contains("saved_vector_slots=0 saved_scalar_slots=0"),
            "{many_dead}"
        );
    }

    #[test]
    fn calls_preserve_crossing_temporaries_without_resaving_spills() {
        let nested = assembly(
            "int id(int x) { if(x<0)return 0; return x; } int main(void) { return id(WARP)+id(WARP+1); }",
        );
        assert_eq!(
            nested.matches("; call id: semantic_live_out=0 []").count(),
            2,
            "{nested}"
        );
        assert!(nested.contains("protected_temporaries=1 [r0]"), "{nested}");
        assert!(nested.contains("saved_vector_slots=1 saved_scalar_slots=0 argument_slots=1 return_slots=1 call_frame_slots=3 call_frame_words=96"), "{nested}");

        let spills = assembly(
            "int id(int x) { if(x<0)return 0; return x; } int main(void) { int a=WARP+1; int b=WARP+2; int c=WARP+3; int d=WARP+4; int e=WARP+5; int f=WARP+6; int g=WARP+7; int h=WARP+8; int i=WARP+9; int j=WARP+10; int r=id(5); return a+b+c+d+e+f+g+h+i+j+r; }",
        );
        assert!(
            spills.contains("semantic_live_out=10 [a, b, c, d, e, f, g, h, i, j]"),
            "{spills}"
        );
        assert!(
            spills.contains("memory_resident=2 [i->frame[0], j->frame[1]]"),
            "{spills}"
        );
        assert!(
            spills.contains("saved_vector_slots=8 saved_scalar_slots=0"),
            "{spills}"
        );
    }

    #[test]
    fn tiny_helpers_inline_without_call_or_callee_ret() {
        let text =
            assembly("int twice(int x) { return x*2; } int main(void) { return twice(21); }");
        assert!(!text.contains("CALL"), "{text}");
        assert!(!text.contains("__warpc_fn_twice:"), "{text}");
        assert!(!text.contains("RET"), "{text}");
    }

    #[test]
    fn inline_bindings_cover_parameters_locals_loops_and_divergence() {
        let text = assembly(
            "int mix(int a,int b) { int sum=a+b; int adjusted=sum+1; return adjusted; } int lane_value(int base) { return base+WARP; } int main(void) { int total=0; for(int i=0;i<2;++i) total+=mix(i,19); if(WARP<16) total+=mix(WARP,1); int v=lane_value(10); return 42+v-v; }",
        );
        assert!(!text.contains("CALL"), "{text}");
        assert!(text.contains("[inlined mix]"));
        assert!(text.contains("[inlined lane_value]"));
        assert!(text.contains("@p3"));
    }

    #[test]
    fn larger_control_flow_helper_keeps_call_ret_abi() {
        let text = assembly(
            "int choose(int x) { if(x<0) return -x; if(x==0) return 1; return x+1; } int main(void) { return choose(41); }",
        );
        assert!(text.contains("CALL __warpc_fn_choose"));
        assert!(text.contains("__warpc_fn_choose:"));
        assert!(text.contains("RET"));
    }

    #[test]
    fn large_straight_line_helper_is_not_silently_expanded() {
        let text = assembly(
            "int large(int x) { int a=x+1; int b=a+1; int c=b+1; int d=c+1; int e=d+1; int f=e+1; int g=f+1; return g; } int main(void) { return large(35); }",
        );
        assert!(text.contains("CALL __warpc_fn_large"));
        assert!(text.contains("__warpc_fn_large:"));
    }

    #[test]
    fn memory_objects_use_word_addressed_load_store() {
        let text = assembly(
            "struct P { int x; char s[3]; }; int main(void) { struct P p; int n=4; int *q=&n; p.s[1]='A'; return *q + p.s[1] + sizeof(p); }",
        );
        assert!(text.contains("MUL r14, r13"));
        assert!(text.contains("LOAD"));
        assert!(text.contains("STORE"));
        assert!(!text.contains("SHL r14, r13, 2"));
    }

    #[test]
    fn divergent_if_uses_nested_execution_masks() {
        let text = assembly(
            "int main(void) { int lane=warp_lane_id(); int x=0; if (lane<16) { if (lane<8) x=42; else x=42; } else { x=42; } return x; }",
        );
        assert!(text.contains("BALLOT p3"));
        assert!(text.contains("ANDMASK p2, p3, p0"));
        assert!(text.contains("@p3"));
        assert!(text.contains("@p2"));
        assert!(text.contains("NOTMASK p3, p3"));
    }

    #[test]
    fn vm_id_is_direct_and_uniform_if_stays_branched() {
        let text = assembly(
            "int main(void) { unsigned vm=warp_vm_id(); int x=0; if (vm==0) x=42; else x=42; return x; }",
        );
        assert!(text.contains("VMID"));
        assert!(text.contains("JMP_IF_ANY"));
        assert!(!text.contains("BALLOT p3"));
    }

    #[test]
    fn conditionals_select_with_branches_or_execution_masks() {
        let uniform =
            assembly("int main(void) { int side=0; return warp_vm_id() ? ++side : (side += 42); }");
        assert!(uniform.contains("conditional_else"));
        assert!(uniform.contains("JMP_IF_ANY"));
        assert!(!uniform.contains("BALLOT p3"));

        let divergent = assembly(
            "int main(void) { int a=1; int b=2; return WARP < 16 ? (a += 3) : (b += 4); }",
        );
        assert!(divergent.contains("BALLOT p3"));
        assert!(divergent.contains("@p3"));
        assert!(divergent.contains("NOTMASK p3, p3"));

        let void_arms =
            assembly("int main(void) { warp_vm_id() ? warp_flip() : warp_flip(); return 42; }");
        assert_eq!(void_arms.matches("FLIP").count(), 2);
        assert!(void_arms.contains("conditional_else"));
    }

    #[test]
    fn min_max_use_existing_unsigned_isa_with_signed_bias() {
        let text = assembly(
            "int main(void) { unsigned u=WARP; int s=WARP-20; return min(s, 7) + max(u, 9u); }",
        );
        assert!(text.contains("MIN r"));
        assert!(text.contains("MAX r"));
        assert!(text.contains(&SIGN_BIT.to_string()));
    }

    #[test]
    fn sequential_local_lifetimes_reuse_physical_homes() {
        let mut source = String::from("int main(void) { int sum=0;");
        for index in 0..24 {
            source.push_str(&format!("int value{index}={index}; sum+=value{index};"));
        }
        source.push_str("return sum; }");
        let text = assembly(&source);
        let summary = text
            .lines()
            .find(|line| line.contains("; allocation main:"))
            .unwrap();
        assert!(summary.contains("spills=0"), "{summary}");
        assert!(summary.contains("scalar_homes=2"), "{summary}");
    }

    #[test]
    fn genuine_temporary_exhaustion_has_an_allocator_diagnostic() {
        let source = "int main(void) { return 1+(2+(3+(4+(5+(6+(7+(8+(9+(10+(11+(12+(13+(14+15))))))))))))); }";
        let tokens = lexer::lex(source).unwrap();
        let ast = parser::parse(&tokens).unwrap();
        let typed = sema::analyze(ast).unwrap();
        let error = generate(&typed).err().unwrap();
        assert!(error.message.contains("temporary vector-register demand"));
        assert!(!error.message.contains("simplify the expression"));
    }
}
