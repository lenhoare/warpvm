use std::mem::discriminant;

use crate::ast::*;
use crate::lexer::{Token, TokenKind};
use crate::span::{Diagnostic, Span};

pub fn parse(tokens: &[Token]) -> Result<Program, Diagnostic> {
    Parser { tokens, current: 0 }.program()
}

struct Parser<'a> {
    tokens: &'a [Token],
    current: usize,
}

impl<'a> Parser<'a> {
    fn program(&mut self) -> Result<Program, Diagnostic> {
        let mut structs = Vec::new();
        let mut globals = Vec::new();
        let mut functions = Vec::new();
        while !self.at(&TokenKind::Eof) {
            if self.is_struct_definition() {
                structs.push(self.struct_definition()?);
                continue;
            }
            let base = self.type_name()?;
            let decl = self.declarator(true)?;
            let Some(name) = decl.name.clone() else {
                unreachable!()
            };
            if self.at(&TokenKind::LParen) && decl.array_len.is_none() {
                functions.push(self.function(base, decl, name)?);
            } else {
                let span = decl.span;
                let init = if self.at(&TokenKind::Equal) {
                    self.bump();
                    Some(self.assignment()?)
                } else {
                    None
                };
                self.expect(
                    TokenKind::Semicolon,
                    "expected ';' after global declaration",
                )?;
                globals.push(Global {
                    ty: DeclType {
                        base,
                        declarator: decl,
                    },
                    init,
                    span,
                });
            }
        }
        if functions.is_empty() {
            return Err(self.error("expected a function declaration"));
        }
        Ok(Program {
            structs,
            globals,
            functions,
        })
    }

    fn is_struct_definition(&self) -> bool {
        self.at(&TokenKind::Struct)
            && matches!(
                self.tokens.get(self.current + 1).map(|t| &t.kind),
                Some(TokenKind::Ident(_))
            )
            && matches!(
                self.tokens.get(self.current + 2).map(|t| &t.kind),
                Some(TokenKind::LBrace)
            )
    }

    fn struct_definition(&mut self) -> Result<StructDecl, Diagnostic> {
        let span = self.bump().span;
        let (name, _) = self.identifier("expected structure name")?;
        self.expect(TokenKind::LBrace, "expected '{' after structure name")?;
        let mut fields = Vec::new();
        while !self.at(&TokenKind::RBrace) {
            let field_span = self.peek().span;
            let base = self.type_name()?;
            let declarator = self.declarator(true)?;
            self.expect(TokenKind::Semicolon, "expected ';' after structure field")?;
            fields.push(FieldDecl {
                ty: DeclType { base, declarator },
                span: field_span,
            });
        }
        self.bump();
        self.expect(
            TokenKind::Semicolon,
            "expected ';' after structure definition",
        )?;
        Ok(StructDecl { name, fields, span })
    }

    fn function(
        &mut self,
        base: TypeName,
        decl: Declarator,
        name: String,
    ) -> Result<Function, Diagnostic> {
        let span = decl.span;
        self.bump();
        let mut params = Vec::new();
        if self.at(&TokenKind::Void)
            && self
                .tokens
                .get(self.current + 1)
                .is_some_and(|t| matches!(t.kind, TokenKind::RParen))
        {
            self.bump();
        } else {
            while !self.at(&TokenKind::RParen) {
                let param_span = self.peek().span;
                let param_base = self.type_name()?;
                let param_decl = self.declarator(false)?;
                params.push(Parameter {
                    ty: DeclType {
                        base: param_base,
                        declarator: param_decl,
                    },
                    span: param_span,
                });
                if !self.at(&TokenKind::Comma) {
                    break;
                }
                self.bump();
            }
        }
        self.expect(TokenKind::RParen, "expected ')' after parameter list")?;
        let body = if self.at(&TokenKind::Semicolon) {
            self.bump();
            None
        } else {
            Some(self.block()?)
        };
        Ok(Function {
            return_type: DeclType {
                base,
                declarator: Declarator {
                    name: None,
                    pointers: decl.pointers,
                    array_len: None,
                    span,
                },
            },
            name,
            params,
            body,
            span,
        })
    }

    fn block(&mut self) -> Result<Block, Diagnostic> {
        let span = self.expect(TokenKind::LBrace, "expected '{'")?;
        let mut statements = Vec::new();
        while !self.at(&TokenKind::RBrace) {
            if self.at(&TokenKind::Eof) {
                return Err(Diagnostic::new(span, "unterminated block"));
            }
            statements.push(if self.starts_type() {
                self.declaration()?
            } else {
                self.statement()?
            });
        }
        self.bump();
        Ok(Block { statements, span })
    }

    fn statement(&mut self) -> Result<Stmt, Diagnostic> {
        if self.at(&TokenKind::If) {
            return self.if_statement();
        }
        if self.at(&TokenKind::While) {
            return self.while_statement();
        }
        if self.at(&TokenKind::Do) {
            return self.do_while_statement();
        }
        if self.at(&TokenKind::For) {
            return self.for_statement();
        }
        if self.at(&TokenKind::Break) || self.at(&TokenKind::Continue) {
            let token = self.bump();
            self.expect(TokenKind::Semicolon, "expected ';' after jump statement")?;
            return Ok(if token.kind == TokenKind::Break {
                Stmt::Break(token.span)
            } else {
                Stmt::Continue(token.span)
            });
        }
        if self.at(&TokenKind::Switch) {
            return self.switch_statement();
        }
        if self.at(&TokenKind::Case) {
            let span = self.bump().span;
            let value = self.expression()?;
            self.expect(TokenKind::Colon, "expected ':' after case value")?;
            return Ok(Stmt::Case {
                value,
                body: Box::new(self.statement()?),
                span,
            });
        }
        if self.at(&TokenKind::Default) {
            let span = self.bump().span;
            self.expect(TokenKind::Colon, "expected ':' after default")?;
            return Ok(Stmt::Default {
                body: Box::new(self.statement()?),
                span,
            });
        }
        if self.at(&TokenKind::Return) {
            let span = self.bump().span;
            let value = if self.at(&TokenKind::Semicolon) {
                None
            } else {
                Some(self.expression()?)
            };
            self.expect(TokenKind::Semicolon, "expected ';' after return")?;
            return Ok(Stmt::Return(value, span));
        }
        if self.at(&TokenKind::LBrace) {
            return Ok(Stmt::Block(self.block()?));
        }
        let span = self.peek().span;
        let value = if self.at(&TokenKind::Semicolon) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(TokenKind::Semicolon, "expected ';' after expression")?;
        Ok(Stmt::Expr(value, span))
    }

    fn parenthesized_condition(&mut self, keyword: &str) -> Result<Expr, Diagnostic> {
        self.expect(TokenKind::LParen, &format!("expected '(' after {keyword}"))?;
        let value = self.expression()?;
        self.expect(TokenKind::RParen, "expected ')' after condition")?;
        Ok(value)
    }

    fn if_statement(&mut self) -> Result<Stmt, Diagnostic> {
        let span = self.bump().span;
        let condition = self.parenthesized_condition("if")?;
        let then_branch = Box::new(self.statement()?);
        let else_branch = if self.at(&TokenKind::Else) {
            self.bump();
            Some(Box::new(self.statement()?))
        } else {
            None
        };
        Ok(Stmt::If {
            condition,
            then_branch,
            else_branch,
            span,
        })
    }

    fn while_statement(&mut self) -> Result<Stmt, Diagnostic> {
        let span = self.bump().span;
        let condition = self.parenthesized_condition("while")?;
        Ok(Stmt::While {
            condition,
            body: Box::new(self.statement()?),
            span,
        })
    }

    fn do_while_statement(&mut self) -> Result<Stmt, Diagnostic> {
        let span = self.bump().span;
        let body = Box::new(self.statement()?);
        self.expect(TokenKind::While, "expected 'while' after do body")?;
        let condition = self.parenthesized_condition("while")?;
        self.expect(TokenKind::Semicolon, "expected ';' after do/while")?;
        Ok(Stmt::DoWhile {
            body,
            condition,
            span,
        })
    }

    fn for_statement(&mut self) -> Result<Stmt, Diagnostic> {
        let span = self.bump().span;
        self.expect(TokenKind::LParen, "expected '(' after for")?;
        let init = if self.at(&TokenKind::Semicolon) {
            self.bump();
            None
        } else if self.starts_type() {
            let decl_span = self.peek().span;
            let base = self.type_name()?;
            let declarator = self.declarator(true)?;
            let init = if self.at(&TokenKind::Equal) {
                self.bump();
                Some(self.assignment()?)
            } else {
                None
            };
            self.expect(TokenKind::Semicolon, "expected ';' after for initializer")?;
            Some(ForInit::Decl {
                ty: DeclType { base, declarator },
                init,
                span: decl_span,
            })
        } else {
            let expr = self.expression()?;
            self.expect(TokenKind::Semicolon, "expected ';' after for initializer")?;
            Some(ForInit::Expr(expr))
        };
        let condition = if self.at(&TokenKind::Semicolon) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(TokenKind::Semicolon, "expected ';' after for condition")?;
        let step = if self.at(&TokenKind::RParen) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(TokenKind::RParen, "expected ')' after for clauses")?;
        Ok(Stmt::For {
            init,
            condition,
            step,
            body: Box::new(self.statement()?),
            span,
        })
    }

    fn switch_statement(&mut self) -> Result<Stmt, Diagnostic> {
        let span = self.bump().span;
        let expression = self.parenthesized_condition("switch")?;
        Ok(Stmt::Switch {
            expression,
            body: Box::new(self.statement()?),
            span,
        })
    }

    fn declaration(&mut self) -> Result<Stmt, Diagnostic> {
        let span = self.peek().span;
        let base = self.type_name()?;
        let declarator = self.declarator(true)?;
        let init = if self.at(&TokenKind::Equal) {
            self.bump();
            Some(self.assignment()?)
        } else {
            None
        };
        self.expect(TokenKind::Semicolon, "expected ';' after declaration")?;
        Ok(Stmt::Decl {
            ty: DeclType { base, declarator },
            init,
            span,
        })
    }

    fn type_name(&mut self) -> Result<TypeName, Diagnostic> {
        let token = self.bump();
        match token.kind {
            TokenKind::Int => Ok(TypeName::Int),
            TokenKind::Unsigned => {
                if self.at(&TokenKind::Int) {
                    self.bump();
                }
                Ok(TypeName::Unsigned)
            }
            TokenKind::CharKw => Ok(TypeName::Char),
            TokenKind::Void => Ok(TypeName::Void),
            TokenKind::Struct => Ok(TypeName::Struct(
                self.identifier("expected structure name")?.0,
            )),
            _ => Err(Diagnostic::new(token.span, "expected Warp C type")),
        }
    }

    fn declarator(&mut self, require_name: bool) -> Result<Declarator, Diagnostic> {
        let span = self.peek().span;
        let mut pointers = 0;
        while self.at(&TokenKind::Star) {
            self.bump();
            pointers += 1;
        }
        let name = if matches!(self.peek().kind, TokenKind::Ident(_)) {
            Some(self.identifier("expected name")?.0)
        } else if require_name {
            return Err(self.error("expected declaration name"));
        } else {
            None
        };
        let array_len = if self.at(&TokenKind::LBracket) {
            self.bump();
            let length = if self.at(&TokenKind::RBracket) {
                None
            } else {
                Some(self.assignment()?)
            };
            self.expect(TokenKind::RBracket, "expected ']' after array size")?;
            Some(length)
        } else {
            None
        };
        Ok(Declarator {
            name,
            pointers,
            array_len,
            span,
        })
    }

    fn expression(&mut self) -> Result<Expr, Diagnostic> {
        let mut expr = self.assignment()?;
        while self.at(&TokenKind::Comma) {
            let span = self.bump().span;
            let right = self.assignment()?;
            expr = Expr {
                kind: ExprKind::Binary(BinaryOp::Comma, Box::new(expr), Box::new(right)),
                span,
            };
        }
        Ok(expr)
    }

    fn assignment(&mut self) -> Result<Expr, Diagnostic> {
        let left = self.logical_or()?;
        let op = match self.peek().kind {
            TokenKind::Equal => AssignOp::Assign,
            TokenKind::PlusEqual => AssignOp::Add,
            TokenKind::MinusEqual => AssignOp::Sub,
            TokenKind::StarEqual => AssignOp::Mul,
            TokenKind::SlashEqual => AssignOp::Div,
            TokenKind::PercentEqual => AssignOp::Mod,
            TokenKind::ShlEqual => AssignOp::Shl,
            TokenKind::ShrEqual => AssignOp::Shr,
            TokenKind::AmpEqual => AssignOp::BitAnd,
            TokenKind::CaretEqual => AssignOp::BitXor,
            TokenKind::PipeEqual => AssignOp::BitOr,
            _ => return Ok(left),
        };
        let span = self.bump().span;
        let right = self.assignment()?;
        Ok(Expr {
            kind: ExprKind::Assign(op, Box::new(left), Box::new(right)),
            span,
        })
    }

    fn logical_or(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(Self::logical_and, &[(TokenKind::OrOr, BinaryOp::LogicalOr)])
    }
    fn logical_and(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(Self::bit_or, &[(TokenKind::AndAnd, BinaryOp::LogicalAnd)])
    }
    fn bit_or(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(Self::bit_xor, &[(TokenKind::Pipe, BinaryOp::BitOr)])
    }
    fn bit_xor(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(Self::bit_and, &[(TokenKind::Caret, BinaryOp::BitXor)])
    }
    fn bit_and(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(Self::equality, &[(TokenKind::Amp, BinaryOp::BitAnd)])
    }
    fn equality(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(
            Self::relational,
            &[
                (TokenKind::EqualEqual, BinaryOp::Eq),
                (TokenKind::BangEqual, BinaryOp::Ne),
            ],
        )
    }
    fn relational(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(
            Self::shift,
            &[
                (TokenKind::Less, BinaryOp::Lt),
                (TokenKind::LessEqual, BinaryOp::Le),
                (TokenKind::Greater, BinaryOp::Gt),
                (TokenKind::GreaterEqual, BinaryOp::Ge),
            ],
        )
    }
    fn shift(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(
            Self::additive,
            &[
                (TokenKind::Shl, BinaryOp::Shl),
                (TokenKind::Shr, BinaryOp::Shr),
            ],
        )
    }
    fn additive(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(
            Self::multiplicative,
            &[
                (TokenKind::Plus, BinaryOp::Add),
                (TokenKind::Minus, BinaryOp::Sub),
            ],
        )
    }
    fn multiplicative(&mut self) -> Result<Expr, Diagnostic> {
        self.left_assoc(
            Self::unary,
            &[
                (TokenKind::Star, BinaryOp::Mul),
                (TokenKind::Slash, BinaryOp::Div),
                (TokenKind::Percent, BinaryOp::Mod),
            ],
        )
    }

    fn left_assoc(
        &mut self,
        next: fn(&mut Self) -> Result<Expr, Diagnostic>,
        ops: &[(TokenKind, BinaryOp)],
    ) -> Result<Expr, Diagnostic> {
        let mut expr = next(self)?;
        loop {
            let Some((_, op)) = ops.iter().find(|(token, _)| self.at(token)) else {
                return Ok(expr);
            };
            let op = *op;
            let span = self.bump().span;
            let right = next(self)?;
            expr = Expr {
                kind: ExprKind::Binary(op, Box::new(expr), Box::new(right)),
                span,
            };
        }
    }

    fn unary(&mut self) -> Result<Expr, Diagnostic> {
        if self.at(&TokenKind::Sizeof) {
            let span = self.bump().span;
            if self.at(&TokenKind::LParen)
                && self
                    .tokens
                    .get(self.current + 1)
                    .is_some_and(|t| Self::kind_starts_type(&t.kind))
            {
                self.bump();
                let base = self.type_name()?;
                let declarator = self.declarator(false)?;
                self.expect(TokenKind::RParen, "expected ')' after sizeof type")?;
                return Ok(Expr {
                    kind: ExprKind::SizeofType(Box::new(DeclType { base, declarator })),
                    span,
                });
            }
            return Ok(Expr {
                kind: ExprKind::SizeofExpr(Box::new(self.unary()?)),
                span,
            });
        }
        let op = match self.peek().kind {
            TokenKind::Plus => UnaryOp::Plus,
            TokenKind::Minus => UnaryOp::Minus,
            TokenKind::Tilde => UnaryOp::BitNot,
            TokenKind::Bang => UnaryOp::LogicalNot,
            TokenKind::Amp => UnaryOp::AddressOf,
            TokenKind::Star => UnaryOp::Deref,
            TokenKind::PlusPlus => UnaryOp::PreInc,
            TokenKind::MinusMinus => UnaryOp::PreDec,
            _ => return self.postfix(),
        };
        let span = self.bump().span;
        Ok(Expr {
            kind: ExprKind::Unary(op, Box::new(self.unary()?)),
            span,
        })
    }

    fn postfix(&mut self) -> Result<Expr, Diagnostic> {
        let mut expr = self.primary()?;
        loop {
            if self.at(&TokenKind::LParen) {
                let span = self.bump().span;
                let mut args = Vec::new();
                while !self.at(&TokenKind::RParen) {
                    args.push(self.assignment()?);
                    if !self.at(&TokenKind::Comma) {
                        break;
                    }
                    self.bump();
                }
                self.expect(TokenKind::RParen, "expected ')' after arguments")?;
                let ExprKind::Name(callee) = expr.kind else {
                    return Err(Diagnostic::new(
                        expr.span,
                        "call target must be a function name",
                    ));
                };
                expr = Expr {
                    kind: ExprKind::Call { callee, args },
                    span,
                };
            } else if self.at(&TokenKind::LBracket) {
                let span = self.bump().span;
                let index = self.expression()?;
                self.expect(TokenKind::RBracket, "expected ']' after index")?;
                expr = Expr {
                    kind: ExprKind::Index(Box::new(expr), Box::new(index)),
                    span,
                };
            } else if self.at(&TokenKind::Dot) || self.at(&TokenKind::Arrow) {
                let token = self.bump();
                let through_pointer = token.kind == TokenKind::Arrow;
                let (field, _) = self.identifier("expected member name")?;
                expr = Expr {
                    kind: ExprKind::Member {
                        base: Box::new(expr),
                        field,
                        through_pointer,
                    },
                    span: token.span,
                };
            } else {
                let op = if self.at(&TokenKind::PlusPlus) {
                    Some(UnaryOp::PostInc)
                } else if self.at(&TokenKind::MinusMinus) {
                    Some(UnaryOp::PostDec)
                } else {
                    None
                };
                let Some(op) = op else { return Ok(expr) };
                let span = self.bump().span;
                expr = Expr {
                    kind: ExprKind::Unary(op, Box::new(expr)),
                    span,
                };
            }
        }
    }

    fn primary(&mut self) -> Result<Expr, Diagnostic> {
        let token = self.bump();
        let kind = match token.kind {
            TokenKind::Number(v) => ExprKind::Number(v),
            TokenKind::Char(v) => ExprKind::Char(v),
            TokenKind::String(v) => ExprKind::String(v),
            TokenKind::Ident(v) => ExprKind::Name(v),
            TokenKind::LParen => {
                let expr = self.expression()?;
                self.expect(TokenKind::RParen, "expected ')' after expression")?;
                return Ok(expr);
            }
            _ => return Err(Diagnostic::new(token.span, "expected expression")),
        };
        Ok(Expr {
            kind,
            span: token.span,
        })
    }

    fn kind_starts_type(kind: &TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Int
                | TokenKind::Unsigned
                | TokenKind::CharKw
                | TokenKind::Void
                | TokenKind::Struct
        )
    }
    fn starts_type(&self) -> bool {
        Self::kind_starts_type(&self.peek().kind)
    }
    fn identifier(&mut self, message: &str) -> Result<(String, Span), Diagnostic> {
        let t = self.bump();
        if let TokenKind::Ident(n) = t.kind {
            Ok((n, t.span))
        } else {
            Err(Diagnostic::new(t.span, message))
        }
    }
    fn expect(&mut self, kind: TokenKind, message: &str) -> Result<Span, Diagnostic> {
        if self.at(&kind) {
            Ok(self.bump().span)
        } else {
            Err(self.error(message))
        }
    }
    fn at(&self, kind: &TokenKind) -> bool {
        discriminant(&self.peek().kind) == discriminant(kind)
    }
    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }
    fn bump(&mut self) -> Token {
        let t = self.tokens[self.current].clone();
        if !matches!(t.kind, TokenKind::Eof) {
            self.current += 1;
        }
        t
    }
    fn error(&self, message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(self.peek().span, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    fn source(text: &str) -> Program {
        parse(&lex(text).unwrap()).unwrap()
    }

    #[test]
    fn precedence_and_associativity() {
        let p = source("int main(void) { int x = 1 + 2 * 3; return x; }");
        let Stmt::Decl {
            init: Some(init), ..
        } = &p.functions[0].body.as_ref().unwrap().statements[0]
        else {
            panic!()
        };
        let ExprKind::Binary(BinaryOp::Add, _, right) = &init.kind else {
            panic!()
        };
        assert!(matches!(right.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
    }

    #[test]
    fn parses_struct_pointer_array_and_string() {
        let p = source("struct Pair { int x; char text[3]; }; int main(void) { struct Pair a; struct Pair *p = &a; char s[] = \"hi\"; p->x = s[1]; return sizeof(a); }");
        assert_eq!(p.structs.len(), 1);
        assert_eq!(p.structs[0].fields.len(), 2);
    }

    #[test]
    fn parses_prototypes_definitions_and_calls() {
        let p = source("int add(int, int); int main(void) { return add(20, 22); } int add(int a, int b) { return a + b; }");
        assert_eq!(p.functions.len(), 3);
        assert!(p.functions[0].body.is_none());
        assert!(p.functions[0].params[0].ty.declarator.name.is_none());
    }
}
