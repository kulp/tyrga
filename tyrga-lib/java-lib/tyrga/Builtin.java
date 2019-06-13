package tyrga;

public class Builtin {
    public static int rem(int a, int b) {
        if (b == 0)
            return a / 0; // defer to division by zero

        // Optimizations
        if (b == 1)
            return 0;

        if (b > 0)
            b = -b;

        // do modulus on negative numbers, because in two's-complement, those
        // can be larger in magnitude than positive numbers can
        if (a > 0)
            return -rem(-a, b);

        while (a <= b)
            a -= b;

        return a;
    }
}

class BuiltinTest {
    private static boolean try_rem_int(int a, int b) {
        int expected = a % b;
        int got = Builtin.rem(a, b);
        return got == expected;
    }

    public static boolean test_rem_int() {
        int smallest = -(1 << 31);
        int largest = (1 << 31) - 1;
        boolean failed = false;
        failed |= ! try_rem_int(smallest, smallest);
        failed |= ! try_rem_int(largest , largest );
        failed |= ! try_rem_int(smallest, largest );
        failed |= ! try_rem_int(largest , smallest);
        if (failed)
            return false;

        for (int a = -100; a < 100; a++) {
            for (int b = -100; b < 100; b++) {
                if (b == 0)
                    continue;
                if (! try_rem_int(a, b)) {
                    System.err.println("a=" + a + ", b=" + b);
                    return false;
                }
            }
        }

        return true;
    }

    public static void main(String[] args) {
        boolean failed = false;

        failed |= !test_rem_int();

        System.exit(failed ? 1 : 0);
    }
}
