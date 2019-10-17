package tyrga;

public class Builtin {
    public static native long add(long a, long b);

    public static long sub(long a, long b) {
        return add(a, -b);
    }

    public static int div(int a, int b) {
        if (b == 0)
            return (1 << 32) - 1; // TODO handle how ?

        // Optimizations
        if (b == 1)
            return a;

        boolean flip = false;
        int c = 0;
        if (b > 0) {
            flip = ! flip;
            b = -b;
        }

        // operate on negative numbers, because in two's-complement, those can
        // be larger in magnitude than positive numbers can
        if (a > 0)
            return -div(-a, b);

        for ( ; a <= b; a -= b)
            c++;

        return flip ? -c : c;
    }

    public static int rem(int a, int b) {
        if (b == 0)
            return a / 0; // defer to division by zero

        if (b > 0)
            b = -b;

        // Optimizations
        if (b == 1)
            return 0;

        // operate on negative numbers, because in two's-complement, those can
        // be larger in magnitude than positive numbers can
        if (a > 0)
            return -rem(-a, b);

        while (a <= b)
            a -= b;

        return a;
    }
}

class BuiltinTest {
    private static boolean try_div_int(int a, int b) {
        int expected = a / b;
        int got = Builtin.div(a, b);
        return got == expected;
    }

    public static boolean test_div_int() {
        int smallest = -(1 << 31);
        int largest = (1 << 31) - 1;
        int cases[] = {
            smallest, smallest + 1, -100, -10, -3, -2, -1, 1, 2, 3, 10, 100, largest - 1, largest
        };
        for (int i = 0; i < cases.length; i++) {
            for (int j = 0; j < cases.length; j++) {
                if (! try_div_int(cases[i], cases[j])) {
                    return false;
                }
            }
        }

        for (int a = -100; a < 100; a++) {
            for (int b = -100; b < 100; b++) {
                if (b == 0)
                    continue;
                if (! try_div_int(a, b)) {
                    System.err.println("a=" + a + ", b=" + b);
                    return false;
                }
            }
        }

        return true;
    }

    private static boolean try_rem_int(int a, int b) {
        int expected = a % b;
        int got = Builtin.rem(a, b);
        return got == expected;
    }

    public static boolean test_rem_int() {
        int smallest = -(1 << 31);
        int largest = (1 << 31) - 1;
        int cases[] = {
            smallest, smallest + 1, -100, -10, -3, -2, -1, 1, 2, 3, 10, 100, largest - 1, largest
        };
        for (int i = 0; i < cases.length; i++) {
            for (int j = 0; j < cases.length; j++) {
                if (! try_rem_int(cases[i], cases[j])) {
                    return false;
                }
            }
        }

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
        failed |= !test_div_int();

        System.exit(failed ? 1 : 0);
    }
}
