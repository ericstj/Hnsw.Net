using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace HnswNet;

/// <summary>
/// Builds the record model for the HNSW connector. HNSW indexes a single vector space, so exactly one
/// vector property is required and multiple vectors are not supported.
/// </summary>
internal sealed class HnswModelBuilder() : CollectionModelBuilder(ValidationOptions)
{
    internal const string SupportedVectorTypes = "ReadOnlyMemory<float>, Embedding<float>, float[]";

    internal static readonly CollectionModelBuildingOptions ValidationOptions = new()
    {
        RequiresAtLeastOneVector = true,
        SupportsMultipleVectors = false,
    };

    protected override bool IsDataPropertyTypeValid(Type type, [NotNullWhen(false)] out string? supportedTypes)
    {
        supportedTypes = null;
        return true;
    }

    protected override bool IsVectorPropertyTypeValid(Type type, [NotNullWhen(false)] out string? supportedTypes)
        => IsVectorPropertyTypeValidCore(type, out supportedTypes);

    internal static bool IsVectorPropertyTypeValidCore(Type type, [NotNullWhen(false)] out string? supportedTypes)
    {
        supportedTypes = SupportedVectorTypes;

        return type == typeof(ReadOnlyMemory<float>)
            || type == typeof(ReadOnlyMemory<float>?)
            || type == typeof(Embedding<float>)
            || type == typeof(float[]);
    }
}
