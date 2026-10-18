using System;
using System.Collections.Generic;
using System.Globalization;

namespace Starlings.Roost.ModelCompiler;

/// <summary>
/// The integer expression language of model definitions: integer literals, names, parentheses and the operators
/// <c>+ - * / %</c> (integer division). Used for shapes, dims and integer or long op arguments, by the generator
/// at compile time and by test evaluators at run time.
/// </summary>
public static class DimExpression
{
    /// <summary>Evaluates <paramref name="text"/>; names are looked up in <paramref name="symbols"/>.</summary>
    /// <param name="text">The expression.</param>
    /// <param name="symbols">Value of each name.</param>
    /// <returns>The value.</returns>
    /// <exception cref="FormatException">The text is not a valid expression, or a name has no value.</exception>
    public static long Evaluate(string text, IReadOnlyDictionary<string, long> symbols)
    {
        if (!TryEvaluate(text, symbols, out long value, out string? error))
        {
            throw new FormatException(error);
        }

        return value;
    }

    /// <summary>Evaluates <paramref name="text"/> without throwing.</summary>
    /// <param name="text">The expression.</param>
    /// <param name="symbols">Value of each name.</param>
    /// <param name="value">The value, when the result is true.</param>
    /// <param name="error">What is wrong, when the result is false.</param>
    /// <returns>True when the expression is valid and every name is known.</returns>
    public static bool TryEvaluate(string text, IReadOnlyDictionary<string, long> symbols, out long value, out string? error)
    {
        var p = new Parser(text, symbols);
        return p.Run(out value, out error);
    }

    /// <summary>The names an expression reads, in order of first use.</summary>
    /// <param name="text">The expression.</param>
    /// <returns>The names; empty when the text has none or is invalid.</returns>
    public static IReadOnlyList<string> Names(string text)
    {
        var names = new List<string>();
        int i = 0;
        while (i < text.Length)
        {
            if (IsNameStart(text[i]))
            {
                int s = i;
                while (i < text.Length && IsNamePart(text[i]))
                {
                    i++;
                }

                string n = text.Substring(s, i - s);
                if (!names.Contains(n))
                {
                    names.Add(n);
                }
            }
            else
            {
                i++;
            }
        }

        return names;
    }

    internal static bool IsNameStart(char c) => char.IsLetter(c) || c == '_';

    internal static bool IsNamePart(char c) => char.IsLetterOrDigit(c) || c == '_';

    private sealed class Parser
    {
        private readonly string _s;
        private readonly IReadOnlyDictionary<string, long> _symbols;
        private int _pos;
        private string? _error;

        public Parser(string s, IReadOnlyDictionary<string, long> symbols)
        {
            _s = s;
            _symbols = symbols;
        }

        public bool Run(out long value, out string? error)
        {
            value = Sum();
            SkipSpace();
            if (_error == null && _pos != _s.Length)
            {
                _error = "unexpected '" + _s[_pos] + "' in '" + _s + "'";
            }

            error = _error;
            return _error == null;
        }

        private void SkipSpace()
        {
            while (_pos < _s.Length && char.IsWhiteSpace(_s[_pos]))
            {
                _pos++;
            }
        }

        private long Sum()
        {
            long v = Product();
            while (_error == null)
            {
                SkipSpace();
                if (_pos < _s.Length && (_s[_pos] == '+' || _s[_pos] == '-'))
                {
                    char op = _s[_pos++];
                    long r = Product();
                    v = op == '+' ? v + r : v - r;
                }
                else
                {
                    break;
                }
            }

            return v;
        }

        private long Product()
        {
            long v = Atom();
            while (_error == null)
            {
                SkipSpace();
                if (_pos < _s.Length && (_s[_pos] == '*' || _s[_pos] == '/' || _s[_pos] == '%'))
                {
                    char op = _s[_pos++];
                    long r = Atom();
                    if (_error != null)
                    {
                        break;
                    }

                    if (op == '*')
                    {
                        v *= r;
                    }
                    else if (r == 0)
                    {
                        _error = "division by zero in '" + _s + "'";
                    }
                    else
                    {
                        v = op == '/' ? v / r : v % r;
                    }
                }
                else
                {
                    break;
                }
            }

            return v;
        }

        private long Atom()
        {
            SkipSpace();
            if (_pos >= _s.Length)
            {
                _error ??= "expression '" + _s + "' ends early";
                return 0;
            }

            char c = _s[_pos];
            if (c == '(')
            {
                _pos++;
                long v = Sum();
                SkipSpace();
                if (_error == null)
                {
                    if (_pos < _s.Length && _s[_pos] == ')')
                    {
                        _pos++;
                    }
                    else
                    {
                        _error = "missing ')' in '" + _s + "'";
                    }
                }

                return v;
            }

            if (c == '-')
            {
                _pos++;
                return -Atom();
            }

            if (char.IsDigit(c))
            {
                int s = _pos;
                while (_pos < _s.Length && char.IsDigit(_s[_pos]))
                {
                    _pos++;
                }

                return long.Parse(_s.Substring(s, _pos - s), CultureInfo.InvariantCulture);
            }

            if (IsNameStart(c))
            {
                int s = _pos;
                while (_pos < _s.Length && IsNamePart(_s[_pos]))
                {
                    _pos++;
                }

                string n = _s.Substring(s, _pos - s);
                if (_symbols.TryGetValue(n, out long v))
                {
                    return v;
                }

                _error = "unknown name '" + n + "' in '" + _s + "'";
                return 0;
            }

            _error = "unexpected '" + c + "' in '" + _s + "'";
            return 0;
        }
    }
}
