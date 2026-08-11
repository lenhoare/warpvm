use std::collections::HashMap;

use crate::ast;
use crate::span::{Diagnostic, Span};

pub type LocalId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Type {
    I32,
    U32,
    Char,
    Void,
}

impl Type {
    pub fn name(self) -> &'static str {
        match self {
            Self::I32 => "int",
            Self::U32 => "unsigned",
            Self::Char => "char",
            Self::Void => "void",
        }
    }

    pub fn is_integer(self) -> bool {
        self != Self::Void
    }

    pub fn is_unsigned(self) -> bool {
        matches!(self, Self::U32 | Self::Char)
    }

    pub fn promoted(self) -> Self {
        if self == Self::Char {
            Self::U32
        } else {
            self
        }
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
pub struct TypedProgram {
    pub return_type: Type,
    pub body: TypedBlock,
    pub locals: Vec<LocalInfo>,
}

#[derive(Clone, Debug)]
pub struct LocalInfo {
    pub name: String,
    pub ty: Type,
    pub span: Span,
    pub uniformity: Uniformity,
}

#[derive(Clone, Debug)]
pub struct TypedBlock {
    pub statements: Vec<TypedStmt>,
    pub local_ids: Vec<LocalId>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypedStmt {
    Decl {
        local: LocalId,
        init: Option<TypedExpr>,
        span: Span,
    },
    Expr(Option<TypedExpr>),
    Return(Option<TypedExpr>, Span),
    Block(TypedBlock),
}

#[derive(Clone, Debug)]
pub struct TypedExpr {
    pub kind: TypedExprKind,
    pub ty: Type,
    pub uniformity: Uniformity,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypedExprKind {
    Literal(u32),
    Local(LocalId),
    Unary(ast::UnaryOp, Box<TypedExpr>),
    Binary {
        op: ast::BinaryOp,
        left: Box<TypedExpr>,
        right: Box<TypedExpr>,
        operand_type: Type,
    },
    Assign {
        local: LocalId,
        op: ast::AssignOp,
        right: Box<TypedExpr>,
        operation_type: Type,
    },
    IncDec {
        local: LocalId,
        increment: bool,
        postfix: bool,
    },
}

pub fn analyze(program: ast::Program) -> Result<TypedProgram, Diagnostic> {
    Analyzer::new().program(program)
}

struct Analyzer {
    scopes: Vec<HashMap<String, LocalId>>,
    locals: Vec<LocalInfo>,
    return_type: Type,
}

impl Analyzer {
    fn new() -> Self {
        Self {
            scopes: Vec::new(),
            locals: Vec::new(),
            return_type: Type::Void,
        }
    }

    fn program(mut self, program: ast::Program) -> Result<TypedProgram, Diagnostic> {
        if program.function.name != "main" {
            return Err(Diagnostic::new(
                program.function.span,
                "the integer slice requires int main(void)",
            ));
        }
        self.return_type = lower_type(program.function.return_type);
        if self.return_type != Type::I32 {
            return Err(Diagnostic::new(
                program.function.span,
                "main must return int",
            ));
        }
        let body = self.block(program.function.body)?;
        Ok(TypedProgram {
            return_type: self.return_type,
            body,
            locals: self.locals,
        })
    }

    fn block(&mut self, block: ast::Block) -> Result<TypedBlock, Diagnostic> {
        self.scopes.push(HashMap::new());
        let mut statements = Vec::new();
        let mut local_ids = Vec::new();
        for statement in block.statements {
            match statement {
                ast::Stmt::Decl {
                    ty,
                    name,
                    init,
                    span,
                } => {
                    let ty = lower_type(ty);
                    if ty == Type::Void {
                        return Err(Diagnostic::new(span, "variable cannot have type void"));
                    }
                    if self.scopes.last().unwrap().contains_key(&name) {
                        return Err(Diagnostic::new(span, format!("duplicate local '{name}'")));
                    }
                    let typed_init = init.map(|expr| self.expr(expr)).transpose()?;
                    if let Some(value) = &typed_init {
                        require_integer(value, "initializer")?;
                    }
                    let uniformity = typed_init
                        .as_ref()
                        .map_or(Uniformity::Uniform, |value| value.uniformity);
                    let id = self.locals.len();
                    self.locals.push(LocalInfo {
                        name: name.clone(),
                        ty,
                        span,
                        uniformity,
                    });
                    self.scopes.last_mut().unwrap().insert(name, id);
                    local_ids.push(id);
                    statements.push(TypedStmt::Decl {
                        local: id,
                        init: typed_init,
                        span,
                    });
                }
                ast::Stmt::Expr(value, _) => {
                    statements.push(TypedStmt::Expr(
                        value.map(|expr| self.expr(expr)).transpose()?,
                    ));
                }
                ast::Stmt::Return(value, span) => {
                    let value = value.map(|expr| self.expr(expr)).transpose()?;
                    if value.is_none() {
                        return Err(Diagnostic::new(span, "non-void main must return a value"));
                    }
                    require_integer(value.as_ref().unwrap(), "return value")?;
                    statements.push(TypedStmt::Return(value, span));
                }
                ast::Stmt::Block(child) => statements.push(TypedStmt::Block(self.block(child)?)),
            }
        }
        self.scopes.pop();
        Ok(TypedBlock {
            statements,
            local_ids,
            span: block.span,
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
                ty: Type::Char,
                uniformity: Uniformity::Uniform,
                span,
            }),
            ast::ExprKind::Name(name) => {
                let local = self
                    .lookup(&name)
                    .ok_or_else(|| Diagnostic::new(span, format!("unknown identifier '{name}'")))?;
                let info = &self.locals[local];
                Ok(TypedExpr {
                    kind: TypedExprKind::Local(local),
                    ty: info.ty,
                    uniformity: info.uniformity,
                    span,
                })
            }
            ast::ExprKind::Unary(op, operand) => self.unary(op, *operand, span),
            ast::ExprKind::Binary(op, left, right) => {
                let left = self.expr(*left)?;
                let right = self.expr(*right)?;
                require_integer(&left, "left operand")?;
                require_integer(&right, "right operand")?;
                let operand_type = usual_type(left.ty, right.ty);
                let ty = match op {
                    ast::BinaryOp::Lt
                    | ast::BinaryOp::Le
                    | ast::BinaryOp::Gt
                    | ast::BinaryOp::Ge
                    | ast::BinaryOp::Eq
                    | ast::BinaryOp::Ne
                    | ast::BinaryOp::LogicalAnd
                    | ast::BinaryOp::LogicalOr => Type::I32,
                    ast::BinaryOp::Shl | ast::BinaryOp::Shr => left.ty.promoted(),
                    ast::BinaryOp::Comma => right.ty,
                    _ => operand_type,
                };
                let uniformity = left.uniformity.join(right.uniformity);
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
            ast::ExprKind::Assign(op, left, right) => {
                let ast::ExprKind::Name(name) = left.kind else {
                    return Err(Diagnostic::new(
                        left.span,
                        "assignment target must be a local variable",
                    ));
                };
                let local = self.lookup(&name).ok_or_else(|| {
                    Diagnostic::new(left.span, format!("unknown identifier '{name}'"))
                })?;
                let right = self.expr(*right)?;
                require_integer(&right, "assignment value")?;
                let local_type = self.locals[local].ty;
                let operation_type = usual_type(local_type, right.ty);
                let uniformity = self.locals[local].uniformity.join(right.uniformity);
                self.locals[local].uniformity = uniformity;
                Ok(TypedExpr {
                    kind: TypedExprKind::Assign {
                        local,
                        op,
                        right: Box::new(right),
                        operation_type,
                    },
                    ty: local_type,
                    uniformity,
                    span,
                })
            }
        }
    }

    fn unary(
        &mut self,
        op: ast::UnaryOp,
        operand: ast::Expr,
        span: Span,
    ) -> Result<TypedExpr, Diagnostic> {
        if matches!(
            op,
            ast::UnaryOp::PreInc
                | ast::UnaryOp::PreDec
                | ast::UnaryOp::PostInc
                | ast::UnaryOp::PostDec
        ) {
            let ast::ExprKind::Name(name) = operand.kind else {
                return Err(Diagnostic::new(
                    operand.span,
                    "increment target must be a local variable",
                ));
            };
            let local = self.lookup(&name).ok_or_else(|| {
                Diagnostic::new(operand.span, format!("unknown identifier '{name}'"))
            })?;
            let info = &self.locals[local];
            return Ok(TypedExpr {
                kind: TypedExprKind::IncDec {
                    local,
                    increment: matches!(op, ast::UnaryOp::PreInc | ast::UnaryOp::PostInc),
                    postfix: matches!(op, ast::UnaryOp::PostInc | ast::UnaryOp::PostDec),
                },
                ty: info.ty,
                uniformity: info.uniformity,
                span,
            });
        }
        let operand = self.expr(operand)?;
        require_integer(&operand, "unary operand")?;
        let ty = if op == ast::UnaryOp::LogicalNot {
            Type::I32
        } else {
            operand.ty.promoted()
        };
        Ok(TypedExpr {
            uniformity: operand.uniformity,
            kind: TypedExprKind::Unary(op, Box::new(operand)),
            ty,
            span,
        })
    }

    fn lookup(&self, name: &str) -> Option<LocalId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }
}

fn lower_type(ty: ast::TypeName) -> Type {
    match ty {
        ast::TypeName::Int => Type::I32,
        ast::TypeName::Unsigned => Type::U32,
        ast::TypeName::Char => Type::Char,
        ast::TypeName::Void => Type::Void,
    }
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
    if expr.ty.is_integer() {
        Ok(())
    } else {
        Err(Diagnostic::new(
            expr.span,
            format!("{role} must have integer type"),
        ))
    }
}

fn parse_integer(text: &str, span: Span) -> Result<(u32, Type), Diagnostic> {
    let compact: String = text.chars().filter(|&ch| ch != '_').collect();
    let (digits, unsigned) = match compact
        .strip_suffix('u')
        .or_else(|| compact.strip_suffix('U'))
    {
        Some(digits) => (digits, true),
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
    let (radix, number) = if let Some(value) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        (16, value)
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
    let ty = if unsigned || value > i32::MAX as u64 {
        Type::U32
    } else {
        Type::I32
    };
    Ok((value as u32, ty))
}

pub fn dump_uniformity(program: &TypedProgram) -> String {
    let mut out = String::new();
    for (id, local) in program.locals.iter().enumerate() {
        out.push_str(&format!(
            "local #{id} {}: {} at {}:{} => {:?}\n",
            local.name,
            local.ty.name(),
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

    fn analyze_source(source: &str) -> Result<TypedProgram, Diagnostic> {
        analyze(parser::parse(&lexer::lex(source)?)?)
    }

    #[test]
    fn int_unsigned_and_char_types() {
        let program = analyze_source(
            "int main(void) { int a = 1; unsigned b = 0xffffffffu; char c = 'A'; return a + b + c; }",
        )
        .unwrap();
        assert_eq!(program.locals[0].ty, Type::I32);
        assert_eq!(program.locals[1].ty, Type::U32);
        assert_eq!(program.locals[2].ty, Type::Char);
    }

    #[test]
    fn rejects_unknown_and_non_lvalue_assignment() {
        let err = analyze_source("int main(void) { 1 = 2; return 0; }").unwrap_err();
        assert!(err.message.contains("assignment target"));
        let err = analyze_source("int main(void) { return missing; }").unwrap_err();
        assert!(err.message.contains("unknown identifier"));
    }
}
