use crate::span::{Diagnostic, Span};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Ident(String),
    Number(String),
    Char(u32),
    Int,
    Unsigned,
    CharKw,
    Void,
    Return,
    If,
    Else,
    While,
    Do,
    For,
    Break,
    Continue,
    Switch,
    Case,
    Default,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Semicolon,
    Comma,
    Colon,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Bang,
    PlusPlus,
    MinusMinus,
    Equal,
    PlusEqual,
    MinusEqual,
    StarEqual,
    SlashEqual,
    PercentEqual,
    AmpEqual,
    PipeEqual,
    CaretEqual,
    Shl,
    Shr,
    ShlEqual,
    ShrEqual,
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    AndAnd,
    OrOr,
    Eof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub fn lex(source: &str) -> Result<Vec<Token>, Diagnostic> {
    Lexer::new(source).run()
}

struct Lexer<'a> {
    source: &'a [u8],
    offset: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            offset: 0,
            line: 1,
            column: 1,
        }
    }

    fn run(mut self) -> Result<Vec<Token>, Diagnostic> {
        let mut tokens = Vec::new();
        loop {
            self.skip_trivia()?;
            let span = self.span();
            let Some(byte) = self.peek(0) else {
                tokens.push(Token {
                    kind: TokenKind::Eof,
                    span,
                });
                return Ok(tokens);
            };
            let kind = if is_ident_start(byte) {
                self.identifier()
            } else if byte.is_ascii_digit() {
                self.number()
            } else if byte == b'\'' {
                TokenKind::Char(self.character()?)
            } else {
                self.operator()?
            };
            tokens.push(Token { kind, span });
        }
    }

    fn span(&self) -> Span {
        Span::new(self.offset, self.line, self.column)
    }

    fn peek(&self, ahead: usize) -> Option<u8> {
        self.source.get(self.offset + ahead).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek(0)?;
        self.offset += 1;
        if byte == b'\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(byte)
    }

    fn skip_trivia(&mut self) -> Result<(), Diagnostic> {
        loop {
            while matches!(self.peek(0), Some(b' ' | b'\t' | b'\r' | b'\n')) {
                self.bump();
            }
            if self.peek(0) == Some(b'/') && self.peek(1) == Some(b'/') {
                while !matches!(self.peek(0), None | Some(b'\n')) {
                    self.bump();
                }
                continue;
            }
            if self.peek(0) == Some(b'/') && self.peek(1) == Some(b'*') {
                let start = self.span();
                self.bump();
                self.bump();
                while !(self.peek(0) == Some(b'*') && self.peek(1) == Some(b'/')) {
                    if self.bump().is_none() {
                        return Err(Diagnostic::new(start, "unterminated block comment"));
                    }
                }
                self.bump();
                self.bump();
                continue;
            }
            return Ok(());
        }
    }

    fn identifier(&mut self) -> TokenKind {
        let start = self.offset;
        while matches!(self.peek(0), Some(byte) if is_ident_continue(byte)) {
            self.bump();
        }
        let text = std::str::from_utf8(&self.source[start..self.offset]).unwrap();
        match text {
            "int" => TokenKind::Int,
            "unsigned" => TokenKind::Unsigned,
            "char" => TokenKind::CharKw,
            "void" => TokenKind::Void,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "do" => TokenKind::Do,
            "for" => TokenKind::For,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "switch" => TokenKind::Switch,
            "case" => TokenKind::Case,
            "default" => TokenKind::Default,
            _ => TokenKind::Ident(text.to_string()),
        }
    }

    fn number(&mut self) -> TokenKind {
        let start = self.offset;
        while matches!(self.peek(0), Some(byte) if byte.is_ascii_alphanumeric() || byte == b'_') {
            self.bump();
        }
        TokenKind::Number(
            std::str::from_utf8(&self.source[start..self.offset])
                .unwrap()
                .to_string(),
        )
    }

    fn character(&mut self) -> Result<u32, Diagnostic> {
        let start = self.span();
        self.bump();
        let value = match self.bump() {
            None | Some(b'\n') => {
                return Err(Diagnostic::new(start, "unterminated character literal"))
            }
            Some(b'\\') => match self.bump() {
                Some(b'n') => b'\n' as u32,
                Some(b'r') => b'\r' as u32,
                Some(b't') => b'\t' as u32,
                Some(b'0') => 0,
                Some(b'\\') => b'\\' as u32,
                Some(b'\'') => b'\'' as u32,
                Some(other) => {
                    return Err(Diagnostic::new(
                        start,
                        format!("unsupported character escape '\\{}'", other as char),
                    ))
                }
                None => return Err(Diagnostic::new(start, "unterminated character escape")),
            },
            Some(byte) if byte.is_ascii() => byte as u32,
            Some(_) => {
                return Err(Diagnostic::new(
                    start,
                    "non-ASCII character literal is not yet supported",
                ))
            }
        };
        if self.bump() != Some(b'\'') {
            return Err(Diagnostic::new(
                start,
                "character literal must contain one character",
            ));
        }
        Ok(value)
    }

    fn operator(&mut self) -> Result<TokenKind, Diagnostic> {
        use TokenKind::*;
        let span = self.span();
        let a = self.bump().unwrap();
        let b = self.peek(0);
        let c = self.peek(1);
        let two = |this: &mut Self, kind| {
            this.bump();
            kind
        };
        let three = |this: &mut Self, kind| {
            this.bump();
            this.bump();
            kind
        };
        let kind = match (a, b, c) {
            (b'<', Some(b'<'), Some(b'=')) => three(self, ShlEqual),
            (b'>', Some(b'>'), Some(b'=')) => three(self, ShrEqual),
            (b'+', Some(b'+'), _) => two(self, PlusPlus),
            (b'-', Some(b'-'), _) => two(self, MinusMinus),
            (b'+', Some(b'='), _) => two(self, PlusEqual),
            (b'-', Some(b'='), _) => two(self, MinusEqual),
            (b'*', Some(b'='), _) => two(self, StarEqual),
            (b'/', Some(b'='), _) => two(self, SlashEqual),
            (b'%', Some(b'='), _) => two(self, PercentEqual),
            (b'&', Some(b'='), _) => two(self, AmpEqual),
            (b'|', Some(b'='), _) => two(self, PipeEqual),
            (b'^', Some(b'='), _) => two(self, CaretEqual),
            (b'=', Some(b'='), _) => two(self, EqualEqual),
            (b'!', Some(b'='), _) => two(self, BangEqual),
            (b'<', Some(b'='), _) => two(self, LessEqual),
            (b'>', Some(b'='), _) => two(self, GreaterEqual),
            (b'&', Some(b'&'), _) => two(self, AndAnd),
            (b'|', Some(b'|'), _) => two(self, OrOr),
            (b'<', Some(b'<'), _) => two(self, Shl),
            (b'>', Some(b'>'), _) => two(self, Shr),
            (b'(', _, _) => LParen,
            (b')', _, _) => RParen,
            (b'{', _, _) => LBrace,
            (b'}', _, _) => RBrace,
            (b';', _, _) => Semicolon,
            (b',', _, _) => Comma,
            (b':', _, _) => Colon,
            (b'+', _, _) => Plus,
            (b'-', _, _) => Minus,
            (b'*', _, _) => Star,
            (b'/', _, _) => Slash,
            (b'%', _, _) => Percent,
            (b'&', _, _) => Amp,
            (b'|', _, _) => Pipe,
            (b'^', _, _) => Caret,
            (b'~', _, _) => Tilde,
            (b'!', _, _) => Bang,
            (b'=', _, _) => Equal,
            (b'<', _, _) => Less,
            (b'>', _, _) => Greater,
            _ => {
                return Err(Diagnostic::new(
                    span,
                    format!("unexpected character '{}'", a as char),
                ))
            }
        };
        Ok(kind)
    }
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_ident_continue(byte: u8) -> bool {
    is_ident_start(byte) || byte.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_literals_and_longest_operators() {
        let tokens = lex("/*a*/ int x = 'A'; // b\nx <<= 2; x != 0;").unwrap();
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Char(65)));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::ShlEqual));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::BangEqual));
    }

    #[test]
    fn reports_unterminated_comment() {
        let err = lex("int x; /*").unwrap_err();
        assert_eq!(err.span.line, 1);
        assert!(err.message.contains("unterminated"));
    }

    #[test]
    fn recognizes_control_flow_keywords_and_colon() {
        let tokens =
            lex("if else while do for break continue switch case 1: default: return").unwrap();
        for kind in [
            TokenKind::If,
            TokenKind::Else,
            TokenKind::While,
            TokenKind::Do,
            TokenKind::For,
            TokenKind::Break,
            TokenKind::Continue,
            TokenKind::Switch,
            TokenKind::Case,
            TokenKind::Default,
            TokenKind::Colon,
        ] {
            assert!(tokens.iter().any(|token| token.kind == kind));
        }
    }
}
