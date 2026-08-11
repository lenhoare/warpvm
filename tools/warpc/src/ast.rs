use crate::span::Span;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeName {
    Int,
    Unsigned,
    Char,
    Void,
    Struct(String),
}

#[derive(Clone, Debug)]
pub struct Declarator {
    pub name: Option<String>,
    pub pointers: usize,
    pub array_len: Option<Option<Expr>>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct DeclType {
    pub base: TypeName,
    pub declarator: Declarator,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub structs: Vec<StructDecl>,
    pub globals: Vec<Global>,
    pub functions: Vec<Function>,
}

#[derive(Clone, Debug)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<FieldDecl>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FieldDecl {
    pub ty: DeclType,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Global {
    pub ty: DeclType,
    pub init: Option<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub return_type: DeclType,
    pub name: String,
    pub params: Vec<Parameter>,
    pub body: Option<Block>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub ty: DeclType,
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
        ty: DeclType,
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
        ty: DeclType,
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
    String(Vec<u32>),
    Name(String),
    Call {
        callee: String,
        args: Vec<Expr>,
    },
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Assign(AssignOp, Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Member {
        base: Box<Expr>,
        field: String,
        through_pointer: bool,
    },
    SizeofExpr(Box<Expr>),
    SizeofType(Box<DeclType>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Plus,
    Minus,
    BitNot,
    LogicalNot,
    AddressOf,
    Deref,
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
