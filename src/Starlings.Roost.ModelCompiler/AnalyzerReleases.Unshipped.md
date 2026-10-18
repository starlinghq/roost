; Unshipped analyzer changes
; https://github.com/dotnet/roslyn-analyzers/blob/main/src/Microsoft.CodeAnalysis.Analyzers/ReleaseTrackingAnalyzers.Help.md

### New Rules

Rule ID | Category | Severity | Notes
--------|----------|----------|-------
ROOST001 | Roost | Error | Op is not a member of IRoostOps
ROOST002 | Roost | Error | Name does not resolve to a value
ROOST003 | Roost | Error | Field is not a member of the op's argument struct
ROOST004 | Roost | Error | Argument binding is not valid for the field
ROOST005 | Roost | Error | IRoostOps is not referenced
ROOST006 | Roost | Error | Shape or dimension is not valid
ROOST007 | Roost | Error | Name is declared twice
ROOST008 | Roost | Error | Model definition is not valid
ROOST009 | Roost | Error | Value is read after an op overwrote it
ROOST010 | Roost | Error | Layer set or placeholder is not valid
ROOST011 | Roost | Error | State group is not valid
ROOST012 | Roost | Error | Fused op has no known unfused form
ROOST013 | Roost | Error | A contract type the generated code needs is not referenced
ROOST014 | Roost | Warning | Weight slot is never read
