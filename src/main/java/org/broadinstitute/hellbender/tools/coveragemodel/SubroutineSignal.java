package org.broadinstitute.hellbender.tools.coveragemodel;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class SubroutineSignal implements Serializable {

    private static final long serialVersionUID = -8990591574704241500L;

    public static final class SubroutineSignalBuilder implements Serializable {

        private static final long serialVersionUID = 1190387682156562190L;

        private final Map<String, Object> result;

        public SubroutineSignalBuilder() {
            result = new HashMap<>();
        }

        public SubroutineSignalBuilder put(final String key, final Object value) {
            result.put(key, value);
            return this;
        }

        public SubroutineSignal build() {
            return new SubroutineSignal(result);
        }
    }

    /* anything the subroutine wishes to communicate */
    private final Map<String, Object> result;

    public SubroutineSignal(final Map<String, Object> result) {
        this.result = result;
    }

    public double getDouble(final String key) {
        return (double)result.get(key);
    }

    public int getInteger(final String key) {
        return (int)result.get(key);
    }

    public String getString(final String key) {
        return (String)result.get(key);
    }

    public INDArray getINDArray(final String key) {
        return (INDArray) result.get(key);
    }

    public boolean getBoolean(final String key) {
        return (boolean)result.get(key);
    }

    public Object getObject(final String key) {
        return result.get(key);
    }

    public static SubroutineSignalBuilder builder() {
        return new SubroutineSignalBuilder();
    }
}