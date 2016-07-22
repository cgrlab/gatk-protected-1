package org.broadinstitute.hellbender.tools.coveragemodel.cachemanager;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Collection;
import java.util.Map;
import java.util.function.Function;

/**
 * This class represents a cache node that can compute
 */
public final class ComputableCacheNode extends CacheNode {

    private final boolean cacheEvals;
    private final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> func;
    private Duplicable cachedValue = null;
    private boolean isCacheCurrent;

    @Override
    public boolean isPrimitive() { return false; }

    @Override
    public boolean isExternallyMutable() { return func == null; }

    /**
     * Constructor
     * @param key the key of the node
     * @param parents immediate parents of the node
     * @param func a function from a map that (at least) contains parents data to the computed value of this node
     * @param cacheEvals does it store the value or not
     */
    public ComputableCacheNode(@Nonnull final String key,
                               @Nonnull final Collection<String> tags,
                               @Nonnull final Collection<String> parents,
                               @Nullable final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> func,
                               final boolean cacheEvals) {
        super(key, tags, parents);
        this.func = func;
        this.cacheEvals = cacheEvals;
        if (func == null && !cacheEvals)
            throw new IllegalArgumentException("A computable node with null evaluation function is externally mutable" +
                    " and must cache its values.");
        isCacheCurrent = false;
    }

    private ComputableCacheNode(@Nonnull final String key,
                                @Nonnull final Collection<String> tags,
                                @Nonnull final Collection<String> parents,
                                @Nullable final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> func,
                                final boolean cacheEvals, final Duplicable cachedValue,
                                final boolean isCacheCurrent) {
        super(key, tags, parents);
        this.func = func;
        this.cacheEvals = cacheEvals;
        this.isCacheCurrent = isCacheCurrent;
        this.cachedValue = cachedValue;
    }

    public boolean isCacheCurrent() { return isCacheCurrent; }

    public boolean cacheEvals() { return cacheEvals; }

    /**
     * Available means (1) the node can store, and (2) a value is available
     * @return boolean
     */
    @Override
    public boolean isStoredValueAvailable() {
        return cacheEvals && cachedValue != null && !cachedValue.isNull();
    }

    /**
     * In addition to being available, this checks if the cached value is up to date
     * @return boolean
     */
    public boolean isStoredValueAvailableAndCurrent() {
        return isStoredValueAvailable() && isCacheCurrent();
    }

    @Override
    public void setValue(@Nullable final Duplicable val) {
        if (!isExternallyMutable()) {
            throw new UnsupportedOperationException("Can not explicitly set the value of a computable cache node with" +
                    " non-null function.");
        } else {
            cachedValue = val;
            isCacheCurrent = true;
        }
    }

    /**
     * Return the value of the node, either from cache, or by computation
     * @param parentsValues a lookup map for parents values
     * @return value
     * @throws IllegalStateException if the node can cache but the cache is not up to date
     */
    @Override
    public Duplicable getValue(@Nonnull final Map<String, ? extends Duplicable> parentsValues) throws IllegalStateException {
        if (cacheEvals) {
            if (isCacheCurrent) {
                return cachedValue;
            } else {
                throw new IllegalStateException("The stored value of the cacheable computable node " + getKey() + " is not current.");
            }
        } else {
            try {
                return func.apply(parentsValues);
            } catch (NullPointerException e) {
                throw new IllegalStateException("Encountered null pointer in computing node (" + getKey() + "). At least " +
                        "one parent node is not property initialized.");
            }
        }
    }

    /**
     * Make a partial clone of the node with a new value; the new value is copied by reference
     * @param newValue the cache value to be replaced with the old value
     * @return
     */
    public ComputableCacheNode duplicateWithUpdatedValue(final Duplicable newValue) {
        if (cacheEvals && newValue != null && !newValue.isNull()) {
            return new ComputableCacheNode(getKey(), getTags(), getParents(), func, true, newValue, true);
        } else {
            return new ComputableCacheNode(getKey(), getTags(), getParents(), func, cacheEvals, null, false);
        }
    }

    /**
     * Returns the underlying function
     * @return function
     */
    public Function<Map<String, ? extends Duplicable>, ? extends Duplicable> getFunction() {
        return func;
    }

    /**
     * Duplicable the cache node with a different status for the stored value
     * @param newCacheUpToDateStatus boolean
     * @return
     */
    public ComputableCacheNode duplicateWithUpdatedCacheStatus(final boolean newCacheUpToDateStatus) {
        return new ComputableCacheNode(getKey(), getTags(), getParents(), func, cacheEvals,
                cachedValue, newCacheUpToDateStatus);
    }

}