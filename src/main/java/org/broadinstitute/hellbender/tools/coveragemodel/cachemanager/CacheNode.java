package org.broadinstitute.hellbender.tools.coveragemodel.cachemanager;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */

import org.broadinstitute.hellbender.utils.Utils;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Collection;
import java.util.Map;

/**
 * This is the base class for cache nodes
 *
 * TODO we would like to make parents and tags unmodifiable, but Kryo throws an exception!
 *
 */
public abstract class CacheNode {

    private final String key;

    private final Collection<String> parents;

    private final Collection<String> tags;

    public abstract Duplicable getValue(@Nonnull final Map<String, ? extends Duplicable> dict)
            throws IllegalStateException;

    public abstract boolean isPrimitive();

    public abstract boolean isStoredValueAvailable();

    public abstract void setValue(@Nullable final Duplicable val);

    public abstract boolean isExternallyMutable();

    public CacheNode duplicateWithUpdatedValue(final Duplicable newValue)
            throws UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }

    public CacheNode duplicate()
            throws UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }

    public CacheNode(@Nonnull final String key,
                     @Nonnull final Collection<String> tags,
                     @Nonnull final Collection<String> parents) {
        this.key = Utils.nonNull(key, "The key of a cache node can not be null");
        this.tags = Utils.nonNull(tags,
                "The allTagsMap of a cache node can not be null");
        this.parents = Utils.nonNull(parents,
                "The immediate parents of a cache node can not be null");
    }

    public String getKey() {
        return key;
    }

    public Collection<String> getParents() {
        return parents;
    }

    public Collection<String> getTags() {
        return tags;
    }

    @Override
    public String toString() {
        return key;
    }
}
