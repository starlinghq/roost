using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

public sealed class CatalogTests
{
    private static byte[] Digest(OpCase c)
    {
        using var ms = new MemoryStream();
        foreach (string name in c.Buffers.Keys.Order(StringComparer.Ordinal))
        {
            ms.Write(c.Buffers[name].Raw());
        }

        foreach (KeyValuePair<string, double> kv in c.Num.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            ms.Write(BitConverter.GetBytes(kv.Value));
        }

        return System.Security.Cryptography.SHA256.HashData(ms.ToArray());
    }

    [Fact]
    public void EveryM0OpWithAVariantHasCases()
    {
        foreach (OpId op in CaseCatalog.Ops)
        {
            var cases = CaseCatalog.Generate(op, 7, CaseProfile.Ci).ToList();
            try
            {
                Assert.NotEmpty(cases);
                Assert.Equal(cases.Count, cases.Select(c => c.Id).Distinct().Count());
                Assert.All(cases, c => Assert.NotEmpty(c.Outputs));
            }
            finally
            {
                cases.ForEach(c => c.Dispose());
            }
        }
    }

    [Fact]
    public void ASeedReproducesTheCasesAndAnotherSeedChangesTheValues()
    {
        foreach (OpId op in CaseCatalog.Ops)
        {
            var a = CaseCatalog.Generate(op, 11, CaseProfile.Ci).ToList();
            var b = CaseCatalog.Generate(op, 11, CaseProfile.Ci).ToList();
            var other = CaseCatalog.Generate(op, 12, CaseProfile.Ci).ToList();
            try
            {
                Assert.Equal(a.Select(c => c.Id), b.Select(c => c.Id));
                for (int i = 0; i < a.Count; i++)
                {
                    Assert.Equal(Digest(a[i]), Digest(b[i]));
                }

                var otherById = other.ToDictionary(c => c.Id);
                Assert.Contains(a.Where(c => otherById.ContainsKey(c.Id)), c => !Digest(c).SequenceEqual(Digest(otherById[c.Id])));
            }
            finally
            {
                a.ForEach(c => c.Dispose());
                b.ForEach(c => c.Dispose());
                other.ForEach(c => c.Dispose());
            }
        }
    }

    [Fact]
    public void RandomCasesRecordTheirSeed()
    {
        foreach (OpId op in CaseCatalog.Ops)
        {
            foreach (OpCase c in CaseCatalog.Generate(op, 5, CaseProfile.Ci))
            {
                using (c)
                {
                    Assert.NotEqual(0UL, c.Seed);
                    Assert.Contains(c.Source, new[] { "fixed", "random" });
                }
            }
        }
    }

    [Fact]
    public void FullProfileHasEveryCiCaseAndMore()
    {
        foreach (OpId op in new[] { OpId.PagedAttention, OpId.Gemm })
        {
            var ci = CaseCatalog.Generate(op, 3, CaseProfile.Ci).Select(c => { using (c) { return c.Id; } }).ToList();
            var full = CaseCatalog.Generate(op, 3, CaseProfile.Full).Select(c => { using (c) { return c.Id; } }).ToList();
            Assert.True(full.Count > ci.Count, op.ToString());
        }
    }

    [Fact]
    public void SingleRowFormsExistForTheRowWiseOps()
    {
        foreach (OpId op in new[] { OpId.RmsNorm, OpId.QkNorm, OpId.Rope, OpId.SiluAndMul, OpId.Gemm, OpId.Embedding, OpId.LmHead })
        {
            bool any = false;
            foreach (OpCase c in CaseCatalog.Generate(op, 1, CaseProfile.Ci))
            {
                using (c)
                {
                    if (c.SingleRow is null || c.Rows < 2)
                    {
                        continue;
                    }

                    any = true;
                    using OpCase one = c.SingleRow(c.Rows - 1);
                    Assert.Equal(c.Op, one.Op);
                    Assert.All(c.Outputs, n => Assert.Equal(c.RowWidth[n], (int)one.Buffers[n].Count));
                }
            }

            Assert.True(any, op.ToString());
        }
    }
}
