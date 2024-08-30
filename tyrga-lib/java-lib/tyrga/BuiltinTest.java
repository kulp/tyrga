package tyrga;

import static org.junit.Assert.assertEquals;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

public class BuiltinTest {
    static Random rng = new Random();
    static int RANDOM_SAMPLES = 100;
    static long RANDOM_SEED = 0; /// TODO support a varying seed

    @BeforeAll
    static void setupSeed() {
        // TODO log the seed used:
        rng.setSeed(RANDOM_SEED);
    }

    @TestFactory
    public Stream<DynamicTest> test_neg() {
        Stream<Integer> fixed = Stream.of(Integer.MIN_VALUE + 1, -5, -1, 0, 2, 4, Integer.MAX_VALUE);
        Stream<Integer> random = Stream.generate(rng::nextInt).limit(10);
        return Stream.concat(fixed, random).map(v -> dynamicTest(v.toString(), () -> assertEquals(-v, Builtin.neg(v))));
    }

    @TestFactory
    public Stream<DynamicTest> test_div_int() {
        int smallest = Integer.MIN_VALUE;
        int largest = Integer.MAX_VALUE;
        int divisor = rng.nextInt(1, 100);
        Stream<Integer> fixed = Stream.of(smallest, smallest + 1, -100, -10, -3, -2, -1, 1, 2, 3, 10, 100, largest - 1,
                largest);
        Stream<Integer> random = Stream.generate(rng::nextInt).limit(RANDOM_SAMPLES);
        return Stream.concat(fixed, random)
                .map(v -> dynamicTest(v.toString(), () -> assertEquals(v / divisor, Builtin.div(v, divisor))));
    }

    @Test
    public void test_div_zero() {
        assertEquals(Integer.MAX_VALUE, Builtin.div(10, 0));
    }

    @TestFactory
    public Stream<DynamicTest> test_rem_int() {
        int smallest = Integer.MIN_VALUE;
        int largest = Integer.MAX_VALUE;
        int divisor = rng.nextInt(1, 100);
        Stream<Integer> fixed = Stream.of(smallest, smallest + 1, -100, -10, -3, -2, -1, 1, 2, 3, 10, 100, largest - 1,
                largest);
        Stream<Integer> random = Stream.generate(rng::nextInt).limit(RANDOM_SAMPLES);
        return Stream.concat(fixed, random)
                .map(v -> dynamicTest(v.toString(), () -> assertEquals(v % divisor, Builtin.rem(v, divisor))));
    }
}
