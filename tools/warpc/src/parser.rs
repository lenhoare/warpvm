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
        let return_type = self.type_name()?;
        let (name, span) = self.identifier("expected function name")?;
        self.expect(TokenKind::LParen, "expected '(' after function name")?;
        if self.at(&TokenKind::Void) {
            self.bump();
        } else if !self.at(&TokenKind::RParen) {
            return Err(self.error("v0.1.4 integer slice supports only a void parameter list"));
        }
        self.expect(TokenKind::RParen, "expected ')' after parameter list")?;
        let body = self.block()?;
        self.expect(
            TokenKind::Eof,
            "only one function is supported in the integer slice",
        )?;
        Ok(Program {
            function: Function {
                return_type,
                name,
                body,
                span,
            },
        })
    }

    fn block(&mut self) -> Result<Block, Diagnostic> {
        let span = self.expect(TokenKind::LBrace, "expected '{'")?;
        let mut statements = Vec::new();
        while !self.at(&TokenKind::RBrace) {
            if self.at(&TokenKind::Eof) {
                return Err(Diagnostic::new(span, "unterminated block"));
            }
            statements.push(self.statement()?);
        }
        self.bump();
        Ok(Block { statements, span })
    }

    fn statement(&mut self) -> Result<Stmt, Diagnostic> {
        if self.starts_type() {
            return self.declaration();
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

    fn declaration(&mut self) -> Result<Stmt, Diagnostic> {
        let span = self.peek().span;
        let ty = self.type_name()?;
        let (name, _) = self.identifier("expected variable name")?;
        let init = if self.at(&TokenKind::Equal) {
            self.bump();
            Some(self.assignment()?)
        } else {
            None
        };
        self.expect(TokenKind::Semicolon, "expected ';' after declaration")?;
        Ok(Stmt::Decl {
            ty,
            name,
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
            _ => Err(Diagnostic::new(token.span, "expected Warp C type")),
        }
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
        operators: &[(TokenKind, BinaryOp)],
    ) -> Result<Expr, Diagnostic> {
        let mut expr = next(self)?;
        loop {
            let Some((_, op)) = operators.iter().find(|(token, _)| self.at(token)) else {
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
        let op = match self.peek().kind {
            TokenKind::Plus => UnaryOp::Plus,
            TokenKind::Minus => UnaryOp::Minus,
            TokenKind::Tilde => UnaryOp::BitNot,
            TokenKind::Bang => UnaryOp::LogicalNot,
            TokenKind::PlusPlus => UnaryOp::PreInc,
            TokenKind::MinusMinus => UnaryOp::PreDec,
            _ => return self.postfix(),
        };
        let span = self.bump().span;
        let operand = self.unary()?;
        Ok(Expr {
            kind: ExprKind::Unary(op, Box::new(operand)),
            span,
        })
    }

    fn postfix(&mut self) -> Result<Expr, Diagnostic> {
        let mut expr = self.primary()?;
        loop {
            let op = if self.at(&TokenKind::PlusPlus) {
                UnaryOp::PostInc
            } else if self.at(&TokenKind::MinusMinus) {
                UnaryOp::PostDec
            } else {
                return Ok(expr);
            };
            let span = self.bump().span;
            expr = Expr {
                kind: ExprKind::Unary(op, Box::new(expr)),
                span,
            };
        }
    }

    fn primary(&mut self) -> Result<Expr, Diagnostic> {
        let token = self.bump();
        let kind = match token.kind {
            TokenKind::Number(value) => ExprKind::Number(value),
            TokenKind::Char(value) => ExprKind::Char(value),
            TokenKind::Ident(name) => ExprKind::Name(name),
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

    fn starts_type(&self) -> bool {
        self.at(&TokenKind::Int)
            || self.at(&TokenKind::Unsigned)
            || self.at(&TokenKind::CharKw)
            || self.at(&TokenKind::Void)
    }

    fn identifier(&mut self, message: &str) -> Result<(String, Span), Diagnostic> {
        let token = self.bump();
        match token.kind {
            TokenKind::Ident(name) => Ok((name, token.span)),
            _ => Err(Diagnostic::new(token.span, message)),
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
        let token = self.tokens[self.current].clone();
        if !matches!(token.kind, TokenKind::Eof) {
            self.current += 1;
        }
        token
    }

    fn error(&self, message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(self.peek().span, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn parse_source(source: &str) -> Program {
        parse(&lex(source).unwrap()).unwrap()
    }

    #[test]
    fn precedence_and_associativity() {
        let program =
            parse_source("int main(void) { int x = 1 + 2 * 3; x = x - 1 - 1; return x; }");
        let Stmt::Decl {
            init: Some(init), ..
        } = &program.function.body.statements[0]
        else {
            panic!("missing declaration")
        };
        let ExprKind::Binary(BinaryOp::Add, _, right) = &init.kind else {
            panic!("add is not expression root")
        };
        assert!(matches!(right.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
    }

    #[test]
    fn compound_and_increment_parse() {
        parse_source("int main(void) { unsigned x = 1u; x <<= 3; ++x; x--; return x; }");
    }
}
