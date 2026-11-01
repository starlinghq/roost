using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost;

/// <summary>
/// One generated token's log probability and its most likely alternatives (R-042), in an update's
/// <c>AdditionalProperties</c> under <see cref="RoostChatClient.LogprobsKey"/>. Values are natural logs of the model's raw
/// distribution, before penalties and temperature.
/// </summary>
/// <param name="TokenId">The generated token.</param>
/// <param name="Logprob">Its log probability.</param>
/// <param name="TopLogprobs">The most likely tokens and their log probabilities, most likely first, as many as the request's <c>top_logprobs</c>.</param>
public sealed record RoostTokenLogprob(int TokenId, float Logprob, IReadOnlyList<TokenLogprob> TopLogprobs);
