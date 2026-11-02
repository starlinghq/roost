using System.Globalization;
using System.Text;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost;
using Starlings.Roost.Generation;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Server.Routing;

/// <summary>
/// Renders a conversation to prompt token ids the way <see cref="RoostChatClient"/> does (the model's chat template, then the
/// tokenizer with no special tokens added), for the routes that count or show prompt tokens without generating:
/// <c>POST /tokenize</c> (chat form) and <c>POST /v1/messages/count_tokens</c>. Text messages only, as in the client; a tool
/// message or an author name is refused naming its field.
/// </summary>
internal static class ChatPrompts
{
    /// <summary>Renders the conversation, with <paramref name="instructions"/> as a leading system message when set.</summary>
    public static string Render(RoostModelProfile model, IReadOnlyList<ChatMessage> messages, string? instructions)
    {
        var conversation = new List<RoostChatMessage>();
        if (instructions is { Length: > 0 })
        {
            conversation.Add(RoostChatMessage.System(instructions));
        }

        for (var i = 0; i < messages.Count; i++)
        {
            conversation.Add(Map(messages[i], i));
        }

        if (conversation.Count == 0)
        {
            throw RequestFields.Refuse("messages", "The conversation is empty.");
        }

        try
        {
            return model.ChatTemplate.Render(conversation);
        }
        catch (Exception e) when (e is not OutOfMemoryException and not OperationCanceledException)
        {
            throw new RoostChatException(new RoostError(RoostErrorCode.InvalidOption, "The chat template rejected the conversation: " + e.Message, "messages"), e);
        }
    }

    /// <summary>The prompt token ids of a text, as the client tokenises a rendered prompt unless <paramref name="addSpecialTokens"/> is set.</summary>
    public static int[] Encode(RoostModelProfile model, string text, bool addSpecialTokens = false) =>
        [.. model.Tokenizer.Encode(text, addSpecialTokens).Ids];

    private static RoostChatMessage Map(ChatMessage message, int index)
    {
        string param = string.Create(CultureInfo.InvariantCulture, $"messages[{index}]");
        RoostChatRole role;
        if (message.Role == ChatRole.System)
        {
            role = RoostChatRole.System;
        }
        else if (message.Role == ChatRole.User)
        {
            role = RoostChatRole.User;
        }
        else if (message.Role == ChatRole.Assistant)
        {
            role = RoostChatRole.Assistant;
        }
        else
        {
            throw RequestFields.Refuse(param + ".role", "Tool messages are unsupported: tools are unsupported.");
        }

        if (!string.IsNullOrEmpty(message.AuthorName))
        {
            throw RequestFields.Refuse(param + ".name", "Message author names are unsupported: the chat template has no field for them.");
        }

        var text = new StringBuilder();
        foreach (var content in message.Contents)
        {
            text.Append(content is TextContent t ? t.Text : throw RequestFields.Refuse(param + ".content", "Only text content is accepted."));
        }

        return new RoostChatMessage(role, text.ToString());
    }
}
