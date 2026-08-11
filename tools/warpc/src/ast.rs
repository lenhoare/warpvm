use crate::span::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeName {
    Int,
    Unsigned,
    Char,
    Void,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub function: Function,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub return_type: TypeName,
    pub name: String,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub statements: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Decl {
        ty: TypeName,
        name: String,
        init: Option<Expr>,
        span: Span,
    },
    Expr(Option<Expr>, Span),
    Return(Option<Expr>, Span),
    Block(Block),
    If {
        condition: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
        span: Span,
    },
    While {
        condition: Expr,
        body: Box<Stmt>,
        span: Span,
    },
    DoWhile {
        body: Box<Stmt>,
        condition: Expr,
        span: Span,
    },
    For {
        init: Option<ForInit>,
        condition: Option<Expr>,
        step: Option<Expr>,
        body: Box<Stmt>,
        span: Span,
    },
    Break(Span),
    Continue(Span),
    Switch {
        expression: Expr,
        body: Box<Stmt>,
        span: Span,
    },
    Case {
        value: Expr,
        body: Box<Stmt>,
        span: Span,
    },
    Default {
        body: Box<Stmt>,
        span: Span,
    },
}

#[derive(Clone, Debug)]
pub enum ForInit {
    Decl {
        ty: TypeName,
        name: String,
        init: Option<Expr>,
        span: Span,
    },
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Number(String),
    Char(u32),
    Name(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Assign(AssignOp, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Plus,
    Minus,
    BitNot,
    LogicalNot,
    PreInc,
    PreDec,
    PostInc,
    PostDec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Mul,
    Div,
    Mod,
    Add,
    Sub,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
    LogicalAnd,
    LogicalOr,
    Comma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shl,
    Shr,
    BitAnd,
    BitXor,
    BitOr,
}
