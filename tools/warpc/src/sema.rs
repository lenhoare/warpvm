use std::collections::HashMap;

use crate::ast;
use crate::span::{Diagnostic, Span};

pub type LocalId = usize;
pub type GlobalId = usize;
pub type FunctionId = usize;
pub type StructId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseType {
    I32,
    U32,
    Char,
    Void,
    Struct(StructId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Type {
    pub base: BaseType,
    pub pointers: usize,
    pub array_len: Option<usize>,
}

impl Type {
    pub const I32: Self = Self::scalar(BaseType::I32);
    pub const U32: Self = Self::scalar(BaseType::U32);
    pub const CHAR: Self = Self::scalar(BaseType::Char);
    pub const VOID: Self = Self::scalar(BaseType::Void);

    const fn scalar(base: BaseType) -> Self {
        Self {
            base,
            pointers: 0,
            array_len: None,
        }
    }

    pub fn pointer_to(mut self) -> Self {
        self.array_len = None;
        self.pointers += 1;
        self
    }

    pub fn decay(self) -> Self {
        if self.array_len.is_some() {
            Self {
                array_len: None,
                pointers: self.pointers + 1,
                ..self
            }
        } else {
            self
        }
    }

    pub fn pointee(self) -> Option<Self> {
        if self.pointers == 0 {
            None
        } else {
            Some(Self {
                pointers: self.pointers - 1,
                array_len: None,
                ..self
            })
        }
    }

    pub fn is_integer(self) -> bool {
        self.pointers == 0
            && self.array_len.is_none()
            && matches!(self.base, BaseType::I32 | BaseType::U32 | BaseType::Char)
    }

    pub fn is_unsigned(self) -> bool {
        self.pointers > 0 || matches!(self.base, BaseType::U32 | BaseType::Char)
    }

    pub fn is_pointer(self) -> bool {
        self.pointers > 0 && self.array_len.is_none()
    }
    pub fn is_array(self) -> bool {
        self.array_len.is_some()
    }
    pub fn is_void(self) -> bool {
        self == Self::VOID
    }
    pub fn is_struct(self) -> bool {
        self.pointers == 0 && self.array_len.is_none() && matches!(self.base, BaseType::Struct(_))
    }
    pub fn is_scalar(self) -> bool {
        self.is_integer() || self.is_pointer()
    }

    pub fn promoted(self) -> Self {
        if self == Self::CHAR {
            Self::U32
        } else {
            self
        }
    }

    pub fn name(self, structs: &[StructInfo]) -> String {
        let mut name = match self.base {
            BaseType::I32 => "int".to_string(),
            BaseType::U32 => "unsigned".to_string(),
            BaseType::Char => "char".to_string(),
            BaseType::Void => "void".to_string(),
            BaseType::Struct(id) => format!("struct {}", structs[id].name),
        };
        name.push_str(&"*".repeat(self.pointers));
        if let Some(length) = self.array_len {
            name.push_str(&format!("[{length}]"));
        }
        name
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uniformity {
    Uniform,
    Divergent,
}

impl Uniformity {
    pub fn join(self, other: Self) -> Self {
        if self == Self::Divergent || other == Self::Divergent {
            Self::Divergent
        } else {
            Self::Uniform
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<FieldInfo>,
    pub size: usize,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FieldInfo {
    pub name: String,
    pub ty: Type,
    pub offset: usize,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct GlobalInfo {
    pub name: String,
    pub ty: Type,
    pub address: usize,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TypedProgram {
    pub functions: Vec<TypedFunction>,
    pub function_names: Vec<String>,
    pub main: FunctionId,
    pub locals: Vec<LocalInfo>,
    pub globals: Vec<GlobalInfo>,
    pub structs: Vec<StructInfo>,
    pub data_words: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct TypedFunction {
    pub id: FunctionId,
    pub name: String,
    pub return_type: Type,
    pub params: Vec<LocalId>,
    pub body: TypedBlock,
    pub frame_words: usize,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct LocalInfo {
    pub name: String,
    pub ty: Type,
    pub span: Span,
    pub uniformity: Uniformity,
    pub address_taken: bool,
    pub frame_offset: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct TypedBlock {
    pub statements: Vec<TypedStmt>,
    pub local_ids: Vec<LocalId>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypedInitializer {
    Expr(TypedExpr),
    Words(Vec<u32>),
}

#[derive(Clone, Debug)]
pub enum TypedStmt {
    Decl {
        local: LocalId,
        init: Option<TypedInitializer>,
        span: Span,
    },
    Expr(Option<TypedExpr>),
    Return(Option<TypedExpr>, Span),
    Block(TypedBlock),
    If {
        condition: TypedExpr,
        then_branch: Box<TypedStmt>,
        else_branch: Option<Box<TypedStmt>>,
        span: Span,
    },
    While {
        condition: TypedExpr,
        body: Box<TypedStmt>,
        span: Span,
    },
    DoWhile {
        body: Box<TypedStmt>,
        condition: TypedExpr,
        span: Span,
    },
    For {
        init: Option<Box<TypedForInit>>,
        condition: Option<Box<TypedExpr>>,
        step: Option<Box<TypedExpr>>,
        body: Box<TypedStmt>,
        local_ids: Vec<LocalId>,
        span: Span,
    },
    Break(Span),
    Continue(Span),
    Switch {
        expression: TypedExpr,
        body: Box<TypedStmt>,
        labels: Vec<SwitchLabel>,
        span: Span,
    },
    Case {
        id: usize,
        body: Box<TypedStmt>,
        span: Span,
    },
    Default {
        id: usize,
        body: Box<TypedStmt>,
        span: Span,
    },
}

#[derive(Clone, Debug)]
pub enum TypedForInit {
    Decl {
        local: LocalId,
        init: Option<TypedInitializer>,
        span: Span,
    },
    Expr(TypedExpr),
}

#[derive(Clone, Debug)]
pub struct SwitchLabel {
    pub id: usize,
    pub value: Option<u32>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TypedExpr {
    pub kind: TypedExprKind,
    pub ty: Type,
    pub uniformity: Uniformity,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    LaneId,
    VmId,
    Framebuffer,
    Flip,
    Argb,
    SetPixel,
}

#[derive(Clone, Debug)]
pub enum LValue {
    Local(LocalId),
    Global(GlobalId),
    Deref(Box<TypedExpr>),
    Index {
        base: Box<TypedExpr>,
        index: Box<TypedExpr>,
        scale: usize,
    },
    Member {
        base: Box<LValue>,
        offset: usize,
    },
}

#[derive(Clone, Debug)]
pub enum TypedExprKind {
    Literal(u32),
    StringAddress(usize),
    Intrinsic {
        intrinsic: Intrinsic,
        args: Vec<TypedExpr>,
    },
    LValue(LValue),
    AddressOf(LValue),
    Unary(ast::UnaryOp, Box<TypedExpr>),
    Binary {
        op: ast::BinaryOp,
        left: Box<TypedExpr>,
        right: Box<TypedExpr>,
        operand_type: Type,
    },
    PointerAdd {
        pointer: Box<TypedExpr>,
        index: Box<TypedExpr>,
        scale: usize,
        subtract: bool,
    },
    PointerDiff {
        left: Box<TypedExpr>,
        right: Box<TypedExpr>,
        scale: usize,
    },
    Assign {
        target: LValue,
        op: ast::AssignOp,
        right: Box<TypedExpr>,
        operation_type: Type,
        scale: usize,
    },
    IncDec {
        target: LValue,
        increment: bool,
        postfix: bool,
        scale: usize,
    },
    Call {
        function: FunctionId,
        args: Vec<TypedExpr>,
    },
}

pub fn analyze(program: ast::Program) -> Result<TypedProgram, Diagnostic> {
    Analyzer::new().program(program)
}

#[derive(Clone)]
struct FunctionSymbol {
    name: String,
    return_type: Type,
    params: Vec<Type>,
    defined: bool,
    span: Span,
    divergent_result: bool,
}
struct SwitchContext {
    labels: Vec<SwitchLabel>,
    values: HashMap<u32, Span>,
    default_span: Option<Span>,
}

struct Analyzer {
    scopes: Vec<HashMap<String, LocalId>>,
    locals: Vec<LocalInfo>,
    globals: Vec<GlobalInfo>,
    global_ids: HashMap<String, GlobalId>,
    structs: Vec<StructInfo>,
    struct_ids: HashMap<String, StructId>,
    data_words: Vec<u32>,
    return_type: Type,
    functions: Vec<FunctionSymbol>,
    function_ids: HashMap<String, FunctionId>,
    current_function: Option<FunctionId>,
    call_edges: Vec<Vec<FunctionId>>,
    loop_depth: usize,
    switch_stack: Vec<SwitchContext>,
    next_switch_label: usize,
    divergent_depth: usize,
}

impl Analyzer {
    fn new() -> Self {
        Self {
            scopes: Vec::new(),
            locals: Vec::new(),
            globals: Vec::new(),
            global_ids: HashMap::new(),
            structs: Vec::new(),
            struct_ids: HashMap::new(),
            data_words: Vec::new(),
            return_type: Type::VOID,
            functions: Vec::new(),
            function_ids: HashMap::new(),
            current_function: None,
            call_edges: Vec::new(),
            loop_depth: 0,
            switch_stack: Vec::new(),
            next_switch_label: 0,
            divergent_depth: 0,
        }
    }

    fn program(mut self, program: ast::Program) -> Result<TypedProgram, Diagnostic> {
        self.register_struct_names(&program.structs)?;
        self.layout_structs(&program.structs)?;
        for global in program.globals {
            self.register_global(global)?;
        }
        for function in &program.functions {
            self.register_function(function)?;
        }
        self.summarize_function_divergence(&program.functions);
        self.call_edges.resize(self.functions.len(), Vec::new());
        let Some(&main) = self.function_ids.get("main") else {
            return Err(Diagnostic::new(
                program.functions[0].span,
                "the current frontend requires int main(void)",
            ));
        };
        let main_symbol = &self.functions[main];
        if !main_symbol.defined
            || main_symbol.return_type != Type::I32
            || !main_symbol.params.is_empty()
        {
            return Err(Diagnostic::new(
                main_symbol.span,
                "program entry must be a definition of int main(void)",
            ));
        }

        let mut typed_functions = Vec::new();
        for function in program.functions {
            let Some(body) = function.body else { continue };
            let id = self.function_ids[&function.name];
            self.current_function = Some(id);
            self.divergent_depth = 0;
            self.return_type = self.functions[id].return_type;
            let local_start = self.locals.len();
            let mut bindings = HashMap::new();
            let mut params = Vec::new();
            for (index, parameter) in function.params.into_iter().enumerate() {
                let Some(name) = parameter.ty.declarator.name else {
                    return Err(Diagnostic::new(
                        parameter.span,
                        "function definitions require parameter names",
                    ));
                };
                if bindings.contains_key(&name) {
                    return Err(Diagnostic::new(
                        parameter.span,
                        format!("duplicate parameter '{name}'"),
                    ));
                }
                let local = self.locals.len();
                self.locals.push(LocalInfo {
                    name: name.clone(),
                    ty: self.functions[id].params[index],
                    span: parameter.span,
                    uniformity: Uniformity::Uniform,
                    address_taken: false,
                    frame_offset: None,
                });
                bindings.insert(name, local);
                params.push(local);
            }
            let body = self.block_with_bindings(body, bindings)?;
            let mut frame_words = 0;
            for local in &mut self.locals[local_start..] {
                if local.address_taken || !local.ty.is_scalar() {
                    local.frame_offset = Some(frame_words);
                    frame_words += type_size(local.ty, &self.structs);
                }
            }
            if self.data_words.len() + frame_words * 32 >= 16_384 {
                return Err(Diagnostic::new(
                    function.span,
                    "Warp C data and local stack frame exceed VM RAM",
                ));
            }
            typed_functions.push(TypedFunction {
                id,
                name: function.name,
                return_type: self.return_type,
                params,
                body,
                frame_words,
                span: function.span,
            });
        }
        self.current_function = None;
        self.validate_call_graph()?;
        let function_names = self.functions.iter().map(|f| f.name.clone()).collect();
        Ok(TypedProgram {
            functions: typed_functions,
            function_names,
            main,
            locals: self.locals,
            globals: self.globals,
            structs: self.structs,
            data_words: self.data_words,
        })
    }

    fn register_struct_names(&mut self, decls: &[ast::StructDecl]) -> Result<(), Diagnostic> {
        for decl in decls {
            if self.struct_ids.contains_key(&decl.name) {
                return Err(Diagnostic::new(
                    decl.span,
                    format!("duplicate definition of struct '{}'", decl.name),
                ));
            }
            let id = self.structs.len();
            self.struct_ids.insert(decl.name.clone(), id);
            self.structs.push(StructInfo {
                name: decl.name.clone(),
                fields: Vec::new(),
                size: 0,
                span: decl.span,
            });
        }
        Ok(())
    }

    fn layout_structs(&mut self, decls: &[ast::StructDecl]) -> Result<(), Diagnostic> {
        for decl in decls {
            let id = self.struct_ids[&decl.name];
            let mut fields = Vec::new();
            let mut names = HashMap::new();
            let mut offset = 0;
            for field in &decl.fields {
                let name = field.ty.declarator.name.clone().unwrap();
                if names.insert(name.clone(), ()).is_some() {
                    return Err(Diagnostic::new(
                        field.span,
                        format!("duplicate field '{name}'"),
                    ));
                }
                let ty = self.lower_decl_type(&field.ty, None)?;
                if ty.is_void() || (ty.base == BaseType::Struct(id) && ty.pointers == 0) {
                    return Err(Diagnostic::new(
                        field.span,
                        "structure field has incomplete or void type",
                    ));
                }
                let size = type_size(ty, &self.structs);
                if size == 0 {
                    return Err(Diagnostic::new(
                        field.span,
                        "structure field has incomplete type",
                    ));
                }
                fields.push(FieldInfo {
                    name,
                    ty,
                    offset,
                    span: field.span,
                });
                offset += size;
            }
            if fields.is_empty() {
                return Err(Diagnostic::new(
                    decl.span,
                    "empty structures are not supported",
                ));
            }
            self.structs[id].fields = fields;
            self.structs[id].size = offset;
        }
        Ok(())
    }

    fn register_global(&mut self, global: ast::Global) -> Result<(), Diagnostic> {
        let name = global.ty.declarator.name.clone().unwrap();
        if self.global_ids.contains_key(&name) {
            return Err(Diagnostic::new(
                global.span,
                format!("duplicate global '{name}'"),
            ));
        }
        let inferred = string_words(&global.init).map(Vec::len);
        let ty = self.lower_decl_type(&global.ty, inferred)?;
        if ty.is_void() {
            return Err(Diagnostic::new(
                global.span,
                "global variable cannot have type void",
            ));
        }
        let address = self.data_words.len();
        let size = type_size(ty, &self.structs);
        if address.checked_add(size).is_none_or(|end| end > 16_384) {
            return Err(Diagnostic::new(
                global.span,
                "global data exceeds the 16,384-word VM RAM",
            ));
        }
        self.data_words.resize(address + size, 0);
        if let Some(init) = global.init {
            match (ty.array_len, init.kind) {
                (Some(length), ast::ExprKind::String(words))
                    if ty.base == BaseType::Char && ty.pointers == 0 =>
                {
                    if words.len() > length {
                        return Err(Diagnostic::new(
                            init.span,
                            "string literal is too long for character array",
                        ));
                    }
                    self.data_words[address..address + words.len()].copy_from_slice(&words);
                }
                (Some(_), _) => {
                    return Err(Diagnostic::new(
                        init.span,
                        "only character arrays may currently have aggregate global initializers",
                    ))
                }
                (None, ast::ExprKind::String(words))
                    if ty.is_pointer() && ty.base == BaseType::Char && ty.pointers == 1 =>
                {
                    let literal_address = self.allocate_words(&words);
                    self.data_words[address] = literal_address as u32;
                }
                (None, kind) if ty.is_integer() || ty.is_pointer() => {
                    let expr = ast::Expr {
                        kind,
                        span: init.span,
                    };
                    self.data_words[address] = constant_ast(&expr)?;
                }
                _ => return Err(Diagnostic::new(init.span, "unsupported global initializer")),
            }
        }
        let id = self.globals.len();
        self.global_ids.insert(name.clone(), id);
        self.globals.push(GlobalInfo {
            name,
            ty,
            address,
            span: global.span,
        });
        Ok(())
    }

    fn allocate_words(&mut self, words: &[u32]) -> usize {
        let address = self.data_words.len();
        self.data_words.extend_from_slice(words);
        address
    }

    fn register_function(&mut self, function: &ast::Function) -> Result<(), Diagnostic> {
        if matches!(
            function.name.as_str(),
            "warp_lane_id"
                | "warp_vm_id"
                | "warp_framebuffer"
                | "warp_flip"
                | "warp_argb"
                | "warp_set_pixel"
        ) {
            return Err(Diagnostic::new(
                function.span,
                format!("'{}' is a reserved Warp C intrinsic", function.name),
            ));
        }
        if self.global_ids.contains_key(&function.name) {
            return Err(Diagnostic::new(
                function.span,
                format!(
                    "'{}' is already declared as a global variable",
                    function.name
                ),
            ));
        }
        let return_type = self.lower_decl_type(&function.return_type, None)?;
        if return_type.is_array() || return_type.is_struct() {
            return Err(Diagnostic::new(
                function.span,
                "functions may not return arrays or structs in this slice",
            ));
        }
        let mut params = Vec::new();
        for parameter in &function.params {
            let mut ty = self.lower_decl_type(&parameter.ty, None)?;
            if ty.is_array() {
                ty = ty.decay();
            }
            if ty.is_void() || ty.is_struct() {
                return Err(Diagnostic::new(
                    parameter.span,
                    "parameter must have scalar or pointer type",
                ));
            }
            params.push(ty);
        }
        if params.len() > 4 {
            return Err(Diagnostic::new(
                function.span,
                "the first Warp C ABI supports at most four parameters",
            ));
        }
        if let Some(&id) = self.function_ids.get(&function.name) {
            let symbol = &mut self.functions[id];
            if symbol.return_type != return_type || symbol.params != params {
                return Err(Diagnostic::new(
                    function.span,
                    format!("conflicting declaration of function '{}'", function.name),
                ));
            }
            if function.body.is_some() {
                if symbol.defined {
                    return Err(Diagnostic::new(
                        function.span,
                        format!("duplicate definition of function '{}'", function.name),
                    ));
                }
                symbol.defined = true;
                symbol.span = function.span;
            }
            return Ok(());
        }
        let id = self.functions.len();
        self.function_ids.insert(function.name.clone(), id);
        self.functions.push(FunctionSymbol {
            name: function.name.clone(),
            return_type,
            params,
            defined: function.body.is_some(),
            span: function.span,
            divergent_result: false,
        });
        Ok(())
    }

    fn summarize_function_divergence(&mut self, functions: &[ast::Function]) {
        let mut direct = vec![false; self.functions.len()];
        let mut edges = vec![Vec::new(); self.functions.len()];
        for function in functions {
            let Some(body) = &function.body else { continue };
            let id = self.function_ids[&function.name];
            scan_block_divergence(body, &self.function_ids, &mut direct[id], &mut edges[id]);
        }
        let mut divergent = direct;
        loop {
            let mut changed = false;
            for id in 0..divergent.len() {
                let value = divergent[id] || edges[id].iter().any(|callee| divergent[*callee]);
                if value != divergent[id] {
                    divergent[id] = value;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for (symbol, divergent) in self.functions.iter_mut().zip(divergent) {
            symbol.divergent_result = divergent;
        }
    }

    fn lower_decl_type(
        &self,
        decl: &ast::DeclType,
        inferred_array: Option<usize>,
    ) -> Result<Type, Diagnostic> {
        let base = match &decl.base {
            ast::TypeName::Int => BaseType::I32,
            ast::TypeName::Unsigned => BaseType::U32,
            ast::TypeName::Char => BaseType::Char,
            ast::TypeName::Void => BaseType::Void,
            ast::TypeName::Struct(name) => {
                BaseType::Struct(*self.struct_ids.get(name).ok_or_else(|| {
                    Diagnostic::new(decl.declarator.span, format!("unknown structure '{name}'"))
                })?)
            }
        };
        let array_len = match &decl.declarator.array_len {
            None => None,
            Some(Some(expr)) => {
                let value = constant_ast(expr)? as usize;
                if value == 0 {
                    return Err(Diagnostic::new(expr.span, "array size must be positive"));
                }
                if value > 16_384 {
                    return Err(Diagnostic::new(
                        expr.span,
                        "array exceeds the 16,384-word VM RAM",
                    ));
                }
                Some(value)
            }
            Some(None) => Some(inferred_array.ok_or_else(|| {
                Diagnostic::new(
                    decl.declarator.span,
                    "array size is required unless inferred from a string literal",
                )
            })?),
        };
        if base == BaseType::Void && decl.declarator.pointers == 0 && array_len.is_some() {
            return Err(Diagnostic::new(
                decl.declarator.span,
                "array element type cannot be void",
            ));
        }
        Ok(Type {
            base,
            pointers: decl.declarator.pointers,
            array_len,
        })
    }

    fn block(&mut self, block: ast::Block) -> Result<TypedBlock, Diagnostic> {
        self.block_with_bindings(block, HashMap::new())
    }
    fn block_with_bindings(
        &mut self,
        block: ast::Block,
        bindings: HashMap<String, LocalId>,
    ) -> Result<TypedBlock, Diagnostic> {
        self.scopes.push(bindings);
        let mut statements = Vec::new();
        let mut local_ids = Vec::new();
        for statement in block.statements {
            let typed = self.statement(statement)?;
            if let TypedStmt::Decl { local, .. } = typed {
                local_ids.push(local);
            }
            statements.push(typed);
        }
        self.scopes.pop();
        Ok(TypedBlock {
            statements,
            local_ids,
            span: block.span,
        })
    }

    fn statement(&mut self, statement: ast::Stmt) -> Result<TypedStmt, Diagnostic> {
        match statement {
            ast::Stmt::Decl { ty, init, span } => self.declaration(ty, init, span),
            ast::Stmt::Expr(value, _) => {
                Ok(TypedStmt::Expr(value.map(|e| self.expr(e)).transpose()?))
            }
            ast::Stmt::Return(value, span) => {
                if self.divergent_depth != 0 {
                    return Err(Diagnostic::new(
                        span,
                        "return inside divergent control flow is not supported in Slice E",
                    ));
                }
                let value = value.map(|e| self.expr(e)).transpose()?;
                if self.return_type.is_void() {
                    if value.is_some() {
                        return Err(Diagnostic::new(span, "void function cannot return a value"));
                    }
                } else {
                    let Some(result) = value.as_ref() else {
                        return Err(Diagnostic::new(
                            span,
                            "non-void function must return a value",
                        ));
                    };
                    self.require_assignable(self.return_type, result, "return value")?;
                }
                Ok(TypedStmt::Return(value, span))
            }
            ast::Stmt::Block(child) => Ok(TypedStmt::Block(self.block(child)?)),
            ast::Stmt::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => {
                let condition = self.condition(condition, "if condition")?;
                let masked =
                    condition.uniformity == Uniformity::Divergent || self.divergent_depth != 0;
                if masked && self.divergent_depth >= 2 {
                    return Err(Diagnostic::new(
                        span,
                        "divergent if nesting exceeds the two predicate masks available in Slice E",
                    ));
                }
                if masked {
                    self.divergent_depth += 1;
                }
                let then_branch = self.statement(*then_branch).map(Box::new);
                let else_branch = else_branch
                    .map(|b| self.statement(*b).map(Box::new))
                    .transpose();
                if masked {
                    self.divergent_depth -= 1;
                }
                Ok(TypedStmt::If {
                    condition,
                    then_branch: then_branch?,
                    else_branch: else_branch?,
                    span,
                })
            }
            ast::Stmt::While {
                condition,
                body,
                span,
            } => {
                self.reject_divergent_control(span, "while loop")?;
                let condition = self.uniform_condition(condition, "while condition")?;
                self.loop_depth += 1;
                let body = self.statement(*body);
                self.loop_depth -= 1;
                Ok(TypedStmt::While {
                    condition,
                    body: Box::new(body?),
                    span,
                })
            }
            ast::Stmt::DoWhile {
                body,
                condition,
                span,
            } => {
                self.reject_divergent_control(span, "do/while loop")?;
                self.loop_depth += 1;
                let body = self.statement(*body);
                self.loop_depth -= 1;
                Ok(TypedStmt::DoWhile {
                    body: Box::new(body?),
                    condition: self.uniform_condition(condition, "do/while condition")?,
                    span,
                })
            }
            ast::Stmt::For {
                init,
                condition,
                step,
                body,
                span,
            } => self.for_statement(init, condition, step, *body, span),
            ast::Stmt::Break(span) => {
                self.reject_divergent_control(span, "break")?;
                if self.loop_depth == 0 && self.switch_stack.is_empty() {
                    return Err(Diagnostic::new(
                        span,
                        "break is not inside a loop or switch",
                    ));
                }
                Ok(TypedStmt::Break(span))
            }
            ast::Stmt::Continue(span) => {
                self.reject_divergent_control(span, "continue")?;
                if self.loop_depth == 0 {
                    return Err(Diagnostic::new(span, "continue is not inside a loop"));
                }
                Ok(TypedStmt::Continue(span))
            }
            ast::Stmt::Switch {
                expression,
                body,
                span,
            } => {
                self.reject_divergent_control(span, "switch")?;
                let expression = self.uniform_condition(expression, "switch expression")?;
                self.switch_stack.push(SwitchContext {
                    labels: Vec::new(),
                    values: HashMap::new(),
                    default_span: None,
                });
                let body = self.statement(*body)?;
                let labels = self.switch_stack.pop().unwrap().labels;
                Ok(TypedStmt::Switch {
                    expression,
                    body: Box::new(body),
                    labels,
                    span,
                })
            }
            ast::Stmt::Case { value, body, span } => {
                if self.switch_stack.is_empty() {
                    return Err(Diagnostic::new(span, "case is not inside a switch"));
                }
                let value = self.expr(value)?;
                require_integer(&value, "case value")?;
                let value = constant_u32(&value)?;
                let context = self.switch_stack.last_mut().unwrap();
                if context.values.insert(value, span).is_some() {
                    return Err(Diagnostic::new(
                        span,
                        format!("duplicate case value {value}"),
                    ));
                }
                let id = self.next_switch_label;
                self.next_switch_label += 1;
                context.labels.push(SwitchLabel {
                    id,
                    value: Some(value),
                    span,
                });
                Ok(TypedStmt::Case {
                    id,
                    body: Box::new(self.statement(*body)?),
                    span,
                })
            }
            ast::Stmt::Default { body, span } => {
                if self.switch_stack.is_empty() {
                    return Err(Diagnostic::new(span, "default is not inside a switch"));
                }
                let context = self.switch_stack.last_mut().unwrap();
                if context.default_span.replace(span).is_some() {
                    return Err(Diagnostic::new(span, "duplicate default label"));
                }
                let id = self.next_switch_label;
                self.next_switch_label += 1;
                context.labels.push(SwitchLabel {
                    id,
                    value: None,
                    span,
                });
                Ok(TypedStmt::Default {
                    id,
                    body: Box::new(self.statement(*body)?),
                    span,
                })
            }
        }
    }

    fn declaration(
        &mut self,
        decl: ast::DeclType,
        init: Option<ast::Expr>,
        span: Span,
    ) -> Result<TypedStmt, Diagnostic> {
        let name = decl.declarator.name.clone().unwrap();
        if self.scopes.last().unwrap().contains_key(&name) {
            return Err(Diagnostic::new(span, format!("duplicate local '{name}'")));
        }
        let inferred = string_words(&init).map(Vec::len);
        let ty = self.lower_decl_type(&decl, inferred)?;
        if ty.is_void() {
            return Err(Diagnostic::new(span, "variable cannot have type void"));
        }
        let typed_init = match init {
            Some(ast::Expr {
                kind: ast::ExprKind::String(words),
                span: init_span,
            }) if ty.is_array() && ty.base == BaseType::Char && ty.pointers == 0 => {
                if words.len() > ty.array_len.unwrap() {
                    return Err(Diagnostic::new(
                        init_span,
                        "string literal is too long for character array",
                    ));
                }
                Some(TypedInitializer::Words(words))
            }
            Some(expr) if ty.is_array() || ty.is_struct() => {
                return Err(Diagnostic::new(
                    expr.span,
                    "aggregate initializer is not supported here",
                ))
            }
            Some(expr) => {
                let value = self.expr(expr)?;
                self.require_assignable(ty, &value, "initializer")?;
                Some(TypedInitializer::Expr(value))
            }
            None => None,
        };
        let mut uniformity = match &typed_init {
            Some(TypedInitializer::Expr(v)) => v.uniformity,
            _ => Uniformity::Uniform,
        };
        if self.divergent_depth != 0 {
            uniformity = Uniformity::Divergent;
        }
        let id = self.locals.len();
        self.locals.push(LocalInfo {
            name: name.clone(),
            ty,
            span,
            uniformity,
            address_taken: false,
            frame_offset: None,
        });
        self.scopes.last_mut().unwrap().insert(name, id);
        Ok(TypedStmt::Decl {
            local: id,
            init: typed_init,
            span,
        })
    }

    fn condition(&mut self, expr: ast::Expr, role: &str) -> Result<TypedExpr, Diagnostic> {
        let expr = self.expr(expr)?;
        if !value_type(expr.ty).is_scalar() {
            return Err(Diagnostic::new(
                expr.span,
                format!("{role} must have scalar type"),
            ));
        }
        Ok(expr)
    }

    fn uniform_condition(&mut self, expr: ast::Expr, role: &str) -> Result<TypedExpr, Diagnostic> {
        let expr = self.condition(expr, role)?;
        if expr.uniformity != Uniformity::Uniform {
            return Err(Diagnostic::new(
                expr.span,
                format!("{role} must be uniform in Slice E"),
            ));
        }
        Ok(expr)
    }

    fn reject_divergent_control(&self, span: Span, construct: &str) -> Result<(), Diagnostic> {
        if self.divergent_depth == 0 {
            Ok(())
        } else {
            Err(Diagnostic::new(
                span,
                format!("{construct} inside divergent control flow is not supported in Slice E"),
            ))
        }
    }

    fn for_statement(
        &mut self,
        init: Option<ast::ForInit>,
        condition: Option<ast::Expr>,
        step: Option<ast::Expr>,
        body: ast::Stmt,
        span: Span,
    ) -> Result<TypedStmt, Diagnostic> {
        self.reject_divergent_control(span, "for loop")?;
        self.scopes.push(HashMap::new());
        let mut local_ids = Vec::new();
        let init = match init {
            Some(ast::ForInit::Decl { ty, init, span }) => {
                let decl = self.declaration(ty, init, span)?;
                let TypedStmt::Decl { local, init, span } = decl else {
                    unreachable!()
                };
                local_ids.push(local);
                Some(TypedForInit::Decl { local, init, span })
            }
            Some(ast::ForInit::Expr(expr)) => Some(TypedForInit::Expr(self.expr(expr)?)),
            None => None,
        };
        let condition = condition
            .map(|e| self.uniform_condition(e, "for condition"))
            .transpose()?;
        let step = step.map(|e| self.expr(e)).transpose()?;
        self.loop_depth += 1;
        let body = self.statement(body);
        self.loop_depth -= 1;
        self.scopes.pop();
        Ok(TypedStmt::For {
            init: init.map(Box::new),
            condition: condition.map(Box::new),
            step: step.map(Box::new),
            body: Box::new(body?),
            local_ids,
            span,
        })
    }

    fn expr(&mut self, expr: ast::Expr) -> Result<TypedExpr, Diagnostic> {
        let span = expr.span;
        match expr.kind {
            ast::ExprKind::Number(text) => {
                let (value, ty) = parse_integer(&text, span)?;
                Ok(TypedExpr {
                    kind: TypedExprKind::Literal(value),
                    ty,
                    uniformity: Uniformity::Uniform,
                    span,
                })
            }
            ast::ExprKind::Char(value) => Ok(TypedExpr {
                kind: TypedExprKind::Literal(value),
                ty: Type::CHAR,
                uniformity: Uniformity::Uniform,
                span,
            }),
            ast::ExprKind::String(words) => {
                let address = self.allocate_words(&words);
                Ok(TypedExpr {
                    kind: TypedExprKind::StringAddress(address),
                    ty: Type::CHAR.pointer_to(),
                    uniformity: Uniformity::Uniform,
                    span,
                })
            }
            ast::ExprKind::Name(name) => {
                if let Some(value) = builtin_constant(&name) {
                    Ok(TypedExpr {
                        kind: TypedExprKind::Literal(value),
                        ty: Type::U32,
                        uniformity: Uniformity::Uniform,
                        span,
                    })
                } else if let Some(local) = self.lookup(&name) {
                    let info = &self.locals[local];
                    Ok(TypedExpr {
                        kind: TypedExprKind::LValue(LValue::Local(local)),
                        ty: info.ty,
                        uniformity: info.uniformity,
                        span,
                    })
                } else if let Some(&global) = self.global_ids.get(&name) {
                    Ok(TypedExpr {
                        kind: TypedExprKind::LValue(LValue::Global(global)),
                        ty: self.globals[global].ty,
                        uniformity: Uniformity::Uniform,
                        span,
                    })
                } else {
                    Err(Diagnostic::new(
                        span,
                        format!("unknown identifier '{name}'"),
                    ))
                }
            }
            ast::ExprKind::Call { callee, args } => self.call(callee, args, span),
            ast::ExprKind::Unary(op, operand) => self.unary(op, *operand, span),
            ast::ExprKind::Binary(op, left, right) => self.binary(op, *left, *right, span),
            ast::ExprKind::Assign(op, left, right) => self.assignment(op, *left, *right, span),
            ast::ExprKind::Index(base, index) => self.index(*base, *index, span),
            ast::ExprKind::Member {
                base,
                field,
                through_pointer,
            } => self.member(*base, field, through_pointer, span),
            ast::ExprKind::SizeofExpr(operand) => {
                let operand = self.expr(*operand)?;
                let size = type_size(operand.ty, &self.structs);
                Ok(TypedExpr {
                    kind: TypedExprKind::Literal(size as u32),
                    ty: Type::U32,
                    uniformity: Uniformity::Uniform,
                    span,
                })
            }
            ast::ExprKind::SizeofType(decl) => {
                let ty = self.lower_decl_type(&decl, None)?;
                let size = type_size(ty, &self.structs);
                if size == 0 {
                    return Err(Diagnostic::new(span, "sizeof incomplete type"));
                }
                Ok(TypedExpr {
                    kind: TypedExprKind::Literal(size as u32),
                    ty: Type::U32,
                    uniformity: Uniformity::Uniform,
                    span,
                })
            }
        }
    }

    fn call(
        &mut self,
        callee: String,
        args: Vec<ast::Expr>,
        span: Span,
    ) -> Result<TypedExpr, Diagnostic> {
        let signature = match callee.as_str() {
            "warp_lane_id" => Some((Intrinsic::LaneId, 0, Type::I32, Uniformity::Divergent)),
            "warp_vm_id" => Some((Intrinsic::VmId, 0, Type::U32, Uniformity::Uniform)),
            "warp_framebuffer" => Some((
                Intrinsic::Framebuffer,
                0,
                Type::U32.pointer_to(),
                Uniformity::Uniform,
            )),
            "warp_flip" => Some((Intrinsic::Flip, 0, Type::VOID, Uniformity::Uniform)),
            "warp_argb" => Some((Intrinsic::Argb, 4, Type::U32, Uniformity::Uniform)),
            "warp_set_pixel" => Some((Intrinsic::SetPixel, 3, Type::VOID, Uniformity::Uniform)),
            _ => None,
        };
        if let Some((intrinsic, arity, ty, base_uniformity)) = signature {
            if args.len() != arity {
                return Err(Diagnostic::new(
                    span,
                    format!(
                        "intrinsic '{callee}' expects {arity} arguments but received {}",
                        args.len()
                    ),
                ));
            }
            if intrinsic == Intrinsic::Flip && self.divergent_depth != 0 {
                return Err(Diagnostic::new(
                    span,
                    "warp_flip() is not allowed inside divergent control flow",
                ));
            }
            let mut uniformity = base_uniformity;
            let mut typed_args = Vec::with_capacity(args.len());
            for arg in args {
                let arg = self.expr(arg)?;
                require_integer(&arg, "graphics intrinsic argument")?;
                uniformity = uniformity.join(arg.uniformity);
                typed_args.push(arg);
            }
            return Ok(TypedExpr {
                kind: TypedExprKind::Intrinsic {
                    intrinsic,
                    args: typed_args,
                },
                ty,
                uniformity,
                span,
            });
        }
        if self.divergent_depth != 0 {
            return Err(Diagnostic::new(
                span,
                "function calls inside divergent control flow are not supported in Slice E",
            ));
        }
        let Some(&function) = self.function_ids.get(&callee) else {
            return Err(Diagnostic::new(
                span,
                format!("unknown function '{callee}'"),
            ));
        };
        if callee == "main" {
            return Err(Diagnostic::new(span, "calling main is not supported"));
        }
        let symbol = self.functions[function].clone();
        if args.len() != symbol.params.len() {
            return Err(Diagnostic::new(
                span,
                format!(
                    "function '{callee}' expects {} arguments but received {}",
                    symbol.params.len(),
                    args.len()
                ),
            ));
        }
        let mut typed_args = Vec::new();
        let mut uniformity = if symbol.divergent_result {
            Uniformity::Divergent
        } else {
            Uniformity::Uniform
        };
        for (arg, param) in args.into_iter().zip(&symbol.params) {
            let arg = self.expr(arg)?;
            self.require_assignable(*param, &arg, "function argument")?;
            uniformity = uniformity.join(arg.uniformity);
            typed_args.push(arg);
        }
        let caller = self.current_function.unwrap();
        if !self.call_edges[caller].contains(&function) {
            self.call_edges[caller].push(function);
        }
        Ok(TypedExpr {
            kind: TypedExprKind::Call {
                function,
                args: typed_args,
            },
            ty: symbol.return_type,
            uniformity,
            span,
        })
    }

    fn unary(
        &mut self,
        op: ast::UnaryOp,
        operand: ast::Expr,
        span: Span,
    ) -> Result<TypedExpr, Diagnostic> {
        if op == ast::UnaryOp::AddressOf {
            let operand = self.expr(operand)?;
            let lvalue = take_lvalue(operand.kind, operand.span, "address-of operand")?;
            self.mark_address_taken(&lvalue);
            let ty = if operand.ty.is_array() {
                operand.ty.decay()
            } else {
                operand.ty.pointer_to()
            };
            return Ok(TypedExpr {
                kind: TypedExprKind::AddressOf(lvalue),
                ty,
                uniformity: operand.uniformity,
                span,
            });
        }
        if op == ast::UnaryOp::Deref {
            let operand = self.expr(operand)?;
            let pointer_ty = value_type(operand.ty);
            let Some(ty) = pointer_ty.pointee() else {
                return Err(Diagnostic::new(
                    span,
                    "cannot dereference a non-pointer value",
                ));
            };
            return Ok(TypedExpr {
                kind: TypedExprKind::LValue(LValue::Deref(Box::new(operand.clone()))),
                ty,
                uniformity: operand.uniformity,
                span,
            });
        }
        if matches!(
            op,
            ast::UnaryOp::PreInc
                | ast::UnaryOp::PreDec
                | ast::UnaryOp::PostInc
                | ast::UnaryOp::PostDec
        ) {
            let operand = self.expr(operand)?;
            let ty = value_type(operand.ty);
            if !ty.is_scalar() {
                return Err(Diagnostic::new(
                    span,
                    "increment target must have integer or pointer type",
                ));
            }
            let target = take_lvalue(operand.kind, operand.span, "increment target")?;
            let uniformity = if self.divergent_depth != 0 {
                Uniformity::Divergent
            } else {
                operand.uniformity
            };
            if let LValue::Local(local) = &target {
                self.locals[*local].uniformity = uniformity;
            }
            let scale = if ty.is_pointer() {
                type_size(ty.pointee().unwrap(), &self.structs)
            } else {
                1
            };
            return Ok(TypedExpr {
                kind: TypedExprKind::IncDec {
                    target,
                    increment: matches!(op, ast::UnaryOp::PreInc | ast::UnaryOp::PostInc),
                    postfix: matches!(op, ast::UnaryOp::PostInc | ast::UnaryOp::PostDec),
                    scale,
                },
                ty,
                uniformity,
                span,
            });
        }
        let operand = self.expr(operand)?;
        require_integer(&operand, "unary operand")?;
        let ty = if op == ast::UnaryOp::LogicalNot {
            Type::I32
        } else {
            value_type(operand.ty).promoted()
        };
        Ok(TypedExpr {
            uniformity: operand.uniformity,
            kind: TypedExprKind::Unary(op, Box::new(operand)),
            ty,
            span,
        })
    }

    fn index(
        &mut self,
        base: ast::Expr,
        index: ast::Expr,
        span: Span,
    ) -> Result<TypedExpr, Diagnostic> {
        let base = self.expr(base)?;
        let index = self.expr(index)?;
        require_integer(&index, "array index")?;
        let pointer_ty = value_type(base.ty);
        let Some(ty) = pointer_ty.pointee() else {
            return Err(Diagnostic::new(
                base.span,
                "indexing requires an array or pointer",
            ));
        };
        let scale = type_size(ty, &self.structs);
        let uniformity = base.uniformity.join(index.uniformity);
        Ok(TypedExpr {
            kind: TypedExprKind::LValue(LValue::Index {
                base: Box::new(base),
                index: Box::new(index),
                scale,
            }),
            ty,
            uniformity,
            span,
        })
    }

    fn member(
        &mut self,
        base: ast::Expr,
        field: String,
        through_pointer: bool,
        span: Span,
    ) -> Result<TypedExpr, Diagnostic> {
        let base_expr = self.expr(base)?;
        let (base_lvalue, struct_ty) = if through_pointer {
            let pointer_ty = value_type(base_expr.ty);
            let Some(struct_ty) = pointer_ty.pointee() else {
                return Err(Diagnostic::new(
                    span,
                    "'->' requires a pointer to structure",
                ));
            };
            (LValue::Deref(Box::new(base_expr.clone())), struct_ty)
        } else {
            let ty = base_expr.ty;
            (
                take_lvalue(base_expr.kind.clone(), base_expr.span, "member base")?,
                ty,
            )
        };
        let BaseType::Struct(id) = struct_ty.base else {
            return Err(Diagnostic::new(span, "member access requires a structure"));
        };
        if !struct_ty.is_struct() {
            return Err(Diagnostic::new(
                span,
                "member access requires a structure object",
            ));
        }
        let field_info = self.structs[id]
            .fields
            .iter()
            .find(|f| f.name == field)
            .ok_or_else(|| {
                Diagnostic::new(
                    span,
                    format!("struct '{}' has no field '{field}'", self.structs[id].name),
                )
            })?;
        Ok(TypedExpr {
            kind: TypedExprKind::LValue(LValue::Member {
                base: Box::new(base_lvalue),
                offset: field_info.offset,
            }),
            ty: field_info.ty,
            uniformity: base_expr.uniformity,
            span,
        })
    }

    fn binary(
        &mut self,
        op: ast::BinaryOp,
        left: ast::Expr,
        right: ast::Expr,
        span: Span,
    ) -> Result<TypedExpr, Diagnostic> {
        let left = self.expr(left)?;
        let right = self.expr(right)?;
        let left_ty = value_type(left.ty);
        let right_ty = value_type(right.ty);
        let uniformity = left.uniformity.join(right.uniformity);
        if matches!(op, ast::BinaryOp::LogicalAnd | ast::BinaryOp::LogicalOr)
            && (uniformity == Uniformity::Divergent || self.divergent_depth != 0)
        {
            return Err(Diagnostic::new(
                span,
                "divergent short-circuit expressions are not supported in Slice E",
            ));
        }
        if matches!(op, ast::BinaryOp::Add | ast::BinaryOp::Sub) {
            if left_ty.is_pointer() && right_ty.is_integer() {
                let scale = type_size(left_ty.pointee().unwrap(), &self.structs);
                return Ok(TypedExpr {
                    kind: TypedExprKind::PointerAdd {
                        pointer: Box::new(left),
                        index: Box::new(right),
                        scale,
                        subtract: op == ast::BinaryOp::Sub,
                    },
                    ty: left_ty,
                    uniformity,
                    span,
                });
            }
            if op == ast::BinaryOp::Add && right_ty.is_pointer() && left_ty.is_integer() {
                let scale = type_size(right_ty.pointee().unwrap(), &self.structs);
                return Ok(TypedExpr {
                    kind: TypedExprKind::PointerAdd {
                        pointer: Box::new(right),
                        index: Box::new(left),
                        scale,
                        subtract: false,
                    },
                    ty: right_ty,
                    uniformity,
                    span,
                });
            }
            if op == ast::BinaryOp::Sub && compatible_pointers(left_ty, right_ty) {
                let scale = type_size(left_ty.pointee().unwrap(), &self.structs);
                return Ok(TypedExpr {
                    kind: TypedExprKind::PointerDiff {
                        left: Box::new(left),
                        right: Box::new(right),
                        scale,
                    },
                    ty: Type::I32,
                    uniformity,
                    span,
                });
            }
        }
        if matches!(op, ast::BinaryOp::Eq | ast::BinaryOp::Ne)
            && (left_ty.is_pointer() || right_ty.is_pointer())
        {
            if !(compatible_pointers(left_ty, right_ty)
                || (left_ty.is_pointer() && is_zero_literal(&right))
                || (right_ty.is_pointer() && is_zero_literal(&left)))
            {
                return Err(Diagnostic::new(span, "incompatible pointer comparison"));
            }
            return Ok(TypedExpr {
                kind: TypedExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                    operand_type: Type::U32,
                },
                ty: Type::I32,
                uniformity,
                span,
            });
        }
        require_integer(&left, "left operand")?;
        require_integer(&right, "right operand")?;
        let operand_type = usual_type(left_ty, right_ty);
        let ty = match op {
            ast::BinaryOp::Lt
            | ast::BinaryOp::Le
            | ast::BinaryOp::Gt
            | ast::BinaryOp::Ge
            | ast::BinaryOp::Eq
            | ast::BinaryOp::Ne
            | ast::BinaryOp::LogicalAnd
            | ast::BinaryOp::LogicalOr => Type::I32,
            ast::BinaryOp::Shl | ast::BinaryOp::Shr => left_ty.promoted(),
            ast::BinaryOp::Comma => right_ty,
            _ => operand_type,
        };
        Ok(TypedExpr {
            kind: TypedExprKind::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                operand_type,
            },
            ty,
            uniformity,
            span,
        })
    }

    fn assignment(
        &mut self,
        op: ast::AssignOp,
        left: ast::Expr,
        right: ast::Expr,
        span: Span,
    ) -> Result<TypedExpr, Diagnostic> {
        let left = self.expr(left)?;
        let left_uniformity = left.uniformity;
        let left_ty = value_type(left.ty);
        if !left_ty.is_scalar() {
            return Err(Diagnostic::new(
                left.span,
                "assignment target must be a scalar lvalue",
            ));
        }
        let target = take_lvalue(left.kind, left.span, "assignment target")?;
        let right = self.expr(right)?;
        if op == ast::AssignOp::Assign {
            self.require_assignable(left_ty, &right, "assignment value")?;
        } else if left_ty.is_pointer() {
            if !matches!(op, ast::AssignOp::Add | ast::AssignOp::Sub)
                || !value_type(right.ty).is_integer()
            {
                return Err(Diagnostic::new(
                    span,
                    "pointer compound assignment only supports += or -= an integer",
                ));
            }
        } else {
            require_integer(&right, "assignment value")?;
        }
        let operation_type = if left_ty.is_pointer() {
            Type::U32
        } else {
            usual_type(left_ty, value_type(right.ty))
        };
        let scale = if left_ty.is_pointer() {
            type_size(left_ty.pointee().unwrap(), &self.structs)
        } else {
            1
        };
        let uniformity = if self.divergent_depth != 0 {
            Uniformity::Divergent
        } else if op == ast::AssignOp::Assign {
            right.uniformity
        } else {
            left_uniformity.join(right.uniformity)
        };
        if let LValue::Local(local) = &target {
            self.locals[*local].uniformity = uniformity;
        }
        Ok(TypedExpr {
            kind: TypedExprKind::Assign {
                target,
                op,
                right: Box::new(right.clone()),
                operation_type,
                scale,
            },
            ty: left_ty,
            uniformity,
            span,
        })
    }

    fn require_assignable(
        &self,
        target: Type,
        value: &TypedExpr,
        role: &str,
    ) -> Result<(), Diagnostic> {
        let source = value_type(value.ty);
        if (target.is_integer() && source.is_integer())
            || compatible_pointers(target, source)
            || (target.is_pointer() && is_zero_literal(value))
        {
            Ok(())
        } else {
            Err(Diagnostic::new(
                value.span,
                format!(
                    "{role} has incompatible type (expected {})",
                    target.name(&self.structs)
                ),
            ))
        }
    }

    fn mark_address_taken(&mut self, lvalue: &LValue) {
        match lvalue {
            LValue::Local(id) => self.locals[*id].address_taken = true,
            LValue::Member { base, .. } => self.mark_address_taken(base),
            LValue::Global(_) | LValue::Deref(_) | LValue::Index { .. } => {}
        }
    }

    fn lookup(&self, name: &str) -> Option<LocalId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn validate_call_graph(&self) -> Result<(), Diagnostic> {
        let mut state = vec![0u8; self.functions.len()];
        let mut depths = vec![0usize; self.functions.len()];
        for id in 0..self.functions.len() {
            if self.functions[id].defined {
                let depth = call_depth(
                    id,
                    &self.functions,
                    &self.call_edges,
                    &mut state,
                    &mut depths,
                )?;
                if depth > 8 {
                    return Err(Diagnostic::new(
                        self.functions[id].span,
                        format!(
                            "function '{}' can exceed WarpVM's eight-entry call stack",
                            self.functions[id].name
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn scan_block_divergence(
    block: &ast::Block,
    function_ids: &HashMap<String, FunctionId>,
    direct: &mut bool,
    edges: &mut Vec<FunctionId>,
) {
    for statement in &block.statements {
        scan_statement_divergence(statement, function_ids, direct, edges);
    }
}

fn scan_statement_divergence(
    statement: &ast::Stmt,
    function_ids: &HashMap<String, FunctionId>,
    direct: &mut bool,
    edges: &mut Vec<FunctionId>,
) {
    match statement {
        ast::Stmt::Decl { init, .. } => {
            if let Some(expr) = init {
                scan_expr_divergence(expr, function_ids, direct, edges);
            }
        }
        ast::Stmt::Expr(expr, _) | ast::Stmt::Return(expr, _) => {
            if let Some(expr) = expr {
                scan_expr_divergence(expr, function_ids, direct, edges);
            }
        }
        ast::Stmt::Block(block) => scan_block_divergence(block, function_ids, direct, edges),
        ast::Stmt::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            scan_expr_divergence(condition, function_ids, direct, edges);
            scan_statement_divergence(then_branch, function_ids, direct, edges);
            if let Some(branch) = else_branch {
                scan_statement_divergence(branch, function_ids, direct, edges);
            }
        }
        ast::Stmt::While {
            condition, body, ..
        } => {
            scan_expr_divergence(condition, function_ids, direct, edges);
            scan_statement_divergence(body, function_ids, direct, edges);
        }
        ast::Stmt::DoWhile {
            body, condition, ..
        } => {
            scan_statement_divergence(body, function_ids, direct, edges);
            scan_expr_divergence(condition, function_ids, direct, edges);
        }
        ast::Stmt::For {
            init,
            condition,
            step,
            body,
            ..
        } => {
            if let Some(init) = init {
                match init {
                    ast::ForInit::Decl { init, .. } => {
                        if let Some(expr) = init {
                            scan_expr_divergence(expr, function_ids, direct, edges);
                        }
                    }
                    ast::ForInit::Expr(expr) => {
                        scan_expr_divergence(expr, function_ids, direct, edges);
                    }
                }
            }
            if let Some(expr) = condition {
                scan_expr_divergence(expr, function_ids, direct, edges);
            }
            if let Some(expr) = step {
                scan_expr_divergence(expr, function_ids, direct, edges);
            }
            scan_statement_divergence(body, function_ids, direct, edges);
        }
        ast::Stmt::Switch {
            expression, body, ..
        } => {
            scan_expr_divergence(expression, function_ids, direct, edges);
            scan_statement_divergence(body, function_ids, direct, edges);
        }
        ast::Stmt::Case { value, body, .. } => {
            scan_expr_divergence(value, function_ids, direct, edges);
            scan_statement_divergence(body, function_ids, direct, edges);
        }
        ast::Stmt::Default { body, .. } => {
            scan_statement_divergence(body, function_ids, direct, edges);
        }
        ast::Stmt::Break(_) | ast::Stmt::Continue(_) => {}
    }
}

fn scan_expr_divergence(
    expr: &ast::Expr,
    function_ids: &HashMap<String, FunctionId>,
    direct: &mut bool,
    edges: &mut Vec<FunctionId>,
) {
    match &expr.kind {
        ast::ExprKind::Call { callee, args } => {
            if callee == "warp_lane_id" {
                *direct = true;
            } else if let Some(id) = function_ids.get(callee) {
                if !edges.contains(id) {
                    edges.push(*id);
                }
            }
            for arg in args {
                scan_expr_divergence(arg, function_ids, direct, edges);
            }
        }
        ast::ExprKind::Unary(_, operand) => {
            scan_expr_divergence(operand, function_ids, direct, edges);
        }
        ast::ExprKind::SizeofExpr(_) => {}
        ast::ExprKind::Binary(_, left, right)
        | ast::ExprKind::Assign(_, left, right)
        | ast::ExprKind::Index(left, right) => {
            scan_expr_divergence(left, function_ids, direct, edges);
            scan_expr_divergence(right, function_ids, direct, edges);
        }
        ast::ExprKind::Member { base, .. } => {
            scan_expr_divergence(base, function_ids, direct, edges);
        }
        ast::ExprKind::Number(_)
        | ast::ExprKind::Char(_)
        | ast::ExprKind::String(_)
        | ast::ExprKind::Name(_)
        | ast::ExprKind::SizeofType(_) => {}
    }
}

fn value_type(ty: Type) -> Type {
    ty.decay()
}
fn compatible_pointers(a: Type, b: Type) -> bool {
    a.is_pointer()
        && b.is_pointer()
        && (a == b || a.pointee() == Some(Type::VOID) || b.pointee() == Some(Type::VOID))
}
fn is_zero_literal(expr: &TypedExpr) -> bool {
    matches!(expr.kind, TypedExprKind::Literal(0))
}
fn take_lvalue(kind: TypedExprKind, span: Span, role: &str) -> Result<LValue, Diagnostic> {
    if let TypedExprKind::LValue(value) = kind {
        Ok(value)
    } else {
        Err(Diagnostic::new(span, format!("{role} must be an lvalue")))
    }
}

pub fn type_size(ty: Type, structs: &[StructInfo]) -> usize {
    if let Some(length) = ty.array_len {
        length
            * type_size(
                Type {
                    array_len: None,
                    ..ty
                },
                structs,
            )
    } else if ty.pointers > 0 {
        1
    } else {
        match ty.base {
            BaseType::I32 | BaseType::U32 | BaseType::Char => 1,
            BaseType::Void => 0,
            BaseType::Struct(id) => structs[id].size,
        }
    }
}

fn string_words(expr: &Option<ast::Expr>) -> Option<&Vec<u32>> {
    match expr.as_ref().map(|e| &e.kind) {
        Some(ast::ExprKind::String(words)) => Some(words),
        _ => None,
    }
}

fn call_depth(
    function: FunctionId,
    functions: &[FunctionSymbol],
    edges: &[Vec<FunctionId>],
    state: &mut [u8],
    depths: &mut [usize],
) -> Result<usize, Diagnostic> {
    if state[function] == 1 {
        return Err(Diagnostic::new(
            functions[function].span,
            format!(
                "recursive call cycle involving '{}' is not supported",
                functions[function].name
            ),
        ));
    }
    if state[function] == 2 {
        return Ok(depths[function]);
    }
    state[function] = 1;
    let mut depth = 0;
    for &callee in &edges[function] {
        if !functions[callee].defined {
            return Err(Diagnostic::new(
                functions[callee].span,
                format!(
                    "function '{}' is declared but not defined",
                    functions[callee].name
                ),
            ));
        }
        depth = depth.max(1 + call_depth(callee, functions, edges, state, depths)?);
    }
    state[function] = 2;
    depths[function] = depth;
    Ok(depth)
}

fn usual_type(left: Type, right: Type) -> Type {
    let left = left.promoted();
    let right = right.promoted();
    if left.is_unsigned() || right.is_unsigned() {
        Type::U32
    } else {
        Type::I32
    }
}
fn require_integer(expr: &TypedExpr, role: &str) -> Result<(), Diagnostic> {
    if value_type(expr.ty).is_integer() {
        Ok(())
    } else {
        Err(Diagnostic::new(
            expr.span,
            format!("{role} must have integer type"),
        ))
    }
}

pub const WARP_VIDEO_WIDTH: u32 = 128;
pub const WARP_VIDEO_HEIGHT: u32 = 128;
pub const WARP_VIDEO_WORDS: u32 = WARP_VIDEO_WIDTH * WARP_VIDEO_HEIGHT;
pub const WARP_VIDEO_BASE: u32 = 0x0010_0000;

fn builtin_constant(name: &str) -> Option<u32> {
    match name {
        "WARP_VIDEO_WIDTH" => Some(WARP_VIDEO_WIDTH),
        "WARP_VIDEO_HEIGHT" => Some(WARP_VIDEO_HEIGHT),
        "WARP_VIDEO_WORDS" => Some(WARP_VIDEO_WORDS),
        "WARP_VIDEO_BASE" => Some(WARP_VIDEO_BASE),
        _ => None,
    }
}

fn constant_ast(expr: &ast::Expr) -> Result<u32, Diagnostic> {
    match &expr.kind {
        ast::ExprKind::Number(text) => Ok(parse_integer(text, expr.span)?.0),
        ast::ExprKind::Char(value) => Ok(*value),
        ast::ExprKind::Name(name) => builtin_constant(name)
            .ok_or_else(|| Diagnostic::new(expr.span, "not a constant integer expression")),
        ast::ExprKind::Unary(op, operand) => {
            let value = constant_ast(operand)?;
            match op {
                ast::UnaryOp::Plus => Ok(value),
                ast::UnaryOp::Minus => Ok(0u32.wrapping_sub(value)),
                ast::UnaryOp::BitNot => Ok(!value),
                ast::UnaryOp::LogicalNot => Ok(u32::from(value == 0)),
                _ => Err(Diagnostic::new(
                    expr.span,
                    "not a constant integer expression",
                )),
            }
        }
        ast::ExprKind::Binary(op, left, right) => {
            let left = constant_ast(left)?;
            let right = constant_ast(right)?;
            use ast::BinaryOp::*;
            Ok(match op {
                Add => left.wrapping_add(right),
                Sub => left.wrapping_sub(right),
                Mul => left.wrapping_mul(right),
                Div if right != 0 => left / right,
                Mod if right != 0 => left % right,
                Shl => left.wrapping_shl(right & 31),
                Shr => left >> (right & 31),
                Lt => u32::from(left < right),
                Le => u32::from(left <= right),
                Gt => u32::from(left > right),
                Ge => u32::from(left >= right),
                Eq => u32::from(left == right),
                Ne => u32::from(left != right),
                BitAnd => left & right,
                BitXor => left ^ right,
                BitOr => left | right,
                LogicalAnd => u32::from(left != 0 && right != 0),
                LogicalOr => u32::from(left != 0 || right != 0),
                Comma => right,
                Div | Mod => {
                    return Err(Diagnostic::new(
                        expr.span,
                        "division by zero in constant expression",
                    ))
                }
            })
        }
        _ => Err(Diagnostic::new(
            expr.span,
            "not a constant integer expression",
        )),
    }
}

fn constant_u32(expr: &TypedExpr) -> Result<u32, Diagnostic> {
    use ast::BinaryOp::*;
    use ast::UnaryOp::*;
    match &expr.kind {
        TypedExprKind::Literal(value) => Ok(*value),
        TypedExprKind::Unary(op, operand) => {
            let value = constant_u32(operand)?;
            Ok(match op {
                Plus => value,
                Minus => 0u32.wrapping_sub(value),
                BitNot => !value,
                LogicalNot => u32::from(value == 0),
                _ => {
                    return Err(Diagnostic::new(
                        expr.span,
                        "case value must be a constant integer expression",
                    ))
                }
            })
        }
        TypedExprKind::Binary {
            op,
            left,
            right,
            operand_type,
        } => {
            let left = constant_u32(left)?;
            if *op == LogicalAnd && left == 0 {
                return Ok(0);
            }
            if *op == LogicalOr && left != 0 {
                return Ok(1);
            }
            let right = constant_u32(right)?;
            let signed = *operand_type == Type::I32;
            Ok(match op {
                Add => left.wrapping_add(right),
                Sub => left.wrapping_sub(right),
                Mul => left.wrapping_mul(right),
                Div | Mod if right == 0 => {
                    return Err(Diagnostic::new(expr.span, "division by zero in case value"))
                }
                Div if signed => ((left as i32 as i64) / (right as i32 as i64)) as u32,
                Mod if signed => ((left as i32 as i64) % (right as i32 as i64)) as u32,
                Div => left / right,
                Mod => left % right,
                Shl => left.wrapping_shl(right & 31),
                Shr if signed => ((left as i32) >> (right & 31)) as u32,
                Shr => left >> (right & 31),
                Lt if signed => u32::from((left as i32) < (right as i32)),
                Le if signed => u32::from((left as i32) <= (right as i32)),
                Gt if signed => u32::from((left as i32) > (right as i32)),
                Ge if signed => u32::from((left as i32) >= (right as i32)),
                Lt => u32::from(left < right),
                Le => u32::from(left <= right),
                Gt => u32::from(left > right),
                Ge => u32::from(left >= right),
                Eq => u32::from(left == right),
                Ne => u32::from(left != right),
                BitAnd => left & right,
                BitXor => left ^ right,
                BitOr => left | right,
                LogicalAnd | LogicalOr => u32::from(right != 0),
                Comma => {
                    return Err(Diagnostic::new(
                        expr.span,
                        "comma is not allowed in a case constant expression",
                    ))
                }
            })
        }
        _ => Err(Diagnostic::new(
            expr.span,
            "case value must be a constant integer expression",
        )),
    }
}

fn parse_integer(text: &str, span: Span) -> Result<(u32, Type), Diagnostic> {
    let compact: String = text.chars().filter(|&ch| ch != '_').collect();
    let (digits, unsigned) = match compact
        .strip_suffix('u')
        .or_else(|| compact.strip_suffix('U'))
    {
        Some(d) => (d, true),
        None => (compact.as_str(), false),
    };
    if digits.ends_with(|ch: char| ch.is_ascii_alphabetic())
        && !digits.starts_with("0x")
        && !digits.starts_with("0X")
    {
        return Err(Diagnostic::new(
            span,
            format!("unsupported integer suffix in '{text}'"),
        ));
    }
    let (radix, number) = if let Some(v) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        (16, v)
    } else {
        (10, digits)
    };
    let value = u64::from_str_radix(number, radix)
        .map_err(|_| Diagnostic::new(span, format!("invalid integer literal '{text}'")))?;
    if value > u32::MAX as u64 {
        return Err(Diagnostic::new(
            span,
            format!("integer literal '{text}' exceeds 32 bits"),
        ));
    }
    Ok((
        value as u32,
        if unsigned || value > i32::MAX as u64 {
            Type::U32
        } else {
            Type::I32
        },
    ))
}

pub fn dump_uniformity(program: &TypedProgram) -> String {
    let mut out = String::new();
    for (id, local) in program.locals.iter().enumerate() {
        out.push_str(&format!(
            "local #{id} {}: {} at {}:{} => {:?}\n",
            local.name,
            local.ty.name(&program.structs),
            local.span.line,
            local.span.column,
            local.uniformity
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer, parser};
    fn source(text: &str) -> Result<TypedProgram, Diagnostic> {
        analyze(parser::parse(&lexer::lex(text)?)?)
    }
    #[test]
    fn scalar_types() {
        let p =
            source("int main(void) { int a=1; unsigned b=2u; char c='A'; return a+b+c; }").unwrap();
        assert_eq!(p.locals[0].ty, Type::I32);
        assert_eq!(p.locals[1].ty, Type::U32);
        assert_eq!(p.locals[2].ty, Type::CHAR);
    }
    #[test]
    fn layouts_words_and_marks_address_taken() {
        let p = source("struct P { int x; char s[3]; }; int main(void) { struct P p; int n=4; int *q=&n; p.s[1]='A'; return sizeof(p)+*q; }").unwrap();
        assert_eq!(p.structs[0].size, 4);
        assert_eq!(p.locals[0].frame_offset, Some(0));
        assert!(p.locals[1].address_taken);
    }
    #[test]
    fn literal_is_zero_terminated_words() {
        let p = source(
            "char *message = \"hello\"; int main(void) { char s[] = \"hello\"; return sizeof(s); }",
        )
        .unwrap();
        assert_eq!(p.locals[0].ty.array_len, Some(6));
        assert_eq!(p.globals[0].address, 0);
        assert_eq!(&p.data_words[1..7], &[104, 101, 108, 108, 111, 0]);
    }
    #[test]
    fn rejects_incompatible_pointer_use() {
        let err = source(
            "struct P { int x; }; int main(void) { int *p=0; struct P *q=0; p=q; return 0; }",
        )
        .unwrap_err();
        assert!(err.message.contains("incompatible"));
    }
    #[test]
    fn rejects_bad_dereference() {
        let err = source("int main(void) { int x=1; return *x; }").unwrap_err();
        assert!(err.message.contains("dereference"));
    }
    #[test]
    fn call_depth_and_recursion() {
        let err = source("int f(void); int main(void){return f();} int f(void){return f();}")
            .unwrap_err();
        assert!(err.message.contains("recursive"));
    }

    #[test]
    fn warp_intrinsics_propagate_uniformity() {
        let p = source(
            "int main(void) { unsigned vm=warp_vm_id(); unsigned lane=warp_lane_id(); unsigned mixed=lane+vm+1; int i=0; for (; i<3; ++i) {} if (mixed<32) i=42; return 42; }",
        )
        .unwrap();
        assert_eq!(p.locals[0].uniformity, Uniformity::Uniform);
        assert_eq!(p.locals[1].uniformity, Uniformity::Divergent);
        assert_eq!(p.locals[2].uniformity, Uniformity::Divergent);
        assert_eq!(p.locals[3].uniformity, Uniformity::Divergent);
    }

    #[test]
    fn rejects_unsupported_divergent_control() {
        let err = source(
            "int main(void) { int lane=warp_lane_id(); while (lane<16) ++lane; return 42; }",
        )
        .unwrap_err();
        assert!(err.message.contains("while condition must be uniform"));
        let err = source(
            "int f(void) { return 1; } int main(void) { int lane=warp_lane_id(); int x=0; if (lane<16) x=f(); return 42; }",
        )
        .unwrap_err();
        assert!(err.message.contains("function calls inside divergent"));
    }

    #[test]
    fn divergent_function_summary_crosses_forward_calls() {
        let p = source(
            "int lane_value(void); int main(void) { int x=lane_value(); if (x<16) x=42; else x=42; return x; } int lane_value(void) { return warp_lane_id(); }",
        )
        .unwrap();
        assert_eq!(p.locals[0].uniformity, Uniformity::Divergent);
        let main = p.functions.iter().find(|f| f.id == p.main).unwrap();
        let TypedStmt::If { condition, .. } = &main.body.statements[1] else {
            panic!()
        };
        assert_eq!(condition.uniformity, Uniformity::Divergent);
    }
}
