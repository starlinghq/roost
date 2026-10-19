using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Starlings.Roost.ModelCompiler;

/// <summary>The layer-set grammar and the <c>{i}</c> placeholder of repeated definitions.</summary>
internal static class Layers
{
    /// <summary>Parses 'a:b[%m=r|%m!=r][,term...]' (b excluded) or a single index into ascending layer indices.</summary>
    public static bool TryParse(string text, out int[] layers, out string? error)
    {
        var set = new SortedSet<int>();
        error = null;
        foreach (string raw in text.Split(','))
        {
            string term = raw.Trim();
            if (term.Length == 0)
            {
                error = "empty term in layer set '" + text + "'";
                layers = Array.Empty<int>();
                return false;
            }

            int modulus = 0, remainder = 0;
            bool negate = false;
            int pct = term.IndexOf('%');
            string range = term;
            if (pct >= 0)
            {
                range = term.Substring(0, pct).Trim();
                string filter = term.Substring(pct + 1);
                int eq = filter.IndexOf('=');
                if (eq < 0)
                {
                    error = "filter '%" + filter + "' needs '=r' or '!=r' in layer set '" + text + "'";
                    layers = Array.Empty<int>();
                    return false;
                }

                string left = filter.Substring(0, eq);
                if (left.EndsWith("!", StringComparison.Ordinal))
                {
                    negate = true;
                    left = left.Substring(0, left.Length - 1);
                }

                if (!int.TryParse(left.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out modulus) || modulus <= 0
                    || !int.TryParse(filter.Substring(eq + 1).Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out remainder))
                {
                    error = "bad filter '%" + filter + "' in layer set '" + text + "'";
                    layers = Array.Empty<int>();
                    return false;
                }
            }

            int start, end;
            int colon = range.IndexOf(':');
            if (colon < 0)
            {
                if (!int.TryParse(range, NumberStyles.None, CultureInfo.InvariantCulture, out start))
                {
                    error = "bad layer '" + range + "' in layer set '" + text + "'";
                    layers = Array.Empty<int>();
                    return false;
                }

                end = start + 1;
            }
            else if (!int.TryParse(range.Substring(0, colon).Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out start)
                || !int.TryParse(range.Substring(colon + 1).Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out end)
                || end < start)
            {
                error = "bad range '" + range + "' in layer set '" + text + "'";
                layers = Array.Empty<int>();
                return false;
            }

            for (int i = start; i < end; i++)
            {
                if (modulus == 0 || ((i % modulus) == remainder) != negate)
                {
                    set.Add(i);
                }
            }
        }

        layers = new int[set.Count];
        set.CopyTo(layers);
        return true;
    }

    /// <summary>Replaces {i}, {i+k} and {i-k} by the layer number. Any other brace expression is an error.</summary>
    public static bool TrySubstitute(string text, int layer, out string result, out string? error)
    {
        error = null;
        if (text.IndexOf('{') < 0)
        {
            result = text;
            return true;
        }

        var sb = new StringBuilder();
        int pos = 0;
        while (pos < text.Length)
        {
            char c = text[pos];
            if (c != '{')
            {
                sb.Append(c);
                pos++;
                continue;
            }

            int close = text.IndexOf('}', pos);
            if (close < 0)
            {
                error = "unclosed '{' in '" + text + "'";
                result = text;
                return false;
            }

            string inner = text.Substring(pos + 1, close - pos - 1).Replace(" ", string.Empty);
            int delta = 0;
            bool ok = inner == "i";
            if (!ok && inner.Length > 2 && inner[0] == 'i' && (inner[1] == '+' || inner[1] == '-')
                && int.TryParse(inner.Substring(2), NumberStyles.None, CultureInfo.InvariantCulture, out delta))
            {
                ok = true;
                if (inner[1] == '-')
                {
                    delta = -delta;
                }
            }

            if (!ok)
            {
                error = "'{" + inner + "}' in '" + text + "' is not {i}, {i+k} or {i-k}";
                result = text;
                return false;
            }

            sb.Append((layer + delta).ToString(CultureInfo.InvariantCulture));
            pos = close + 1;
        }

        result = sb.ToString();
        return true;
    }
}
