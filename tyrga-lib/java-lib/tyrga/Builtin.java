package tyrga;

public class Builtin {
    public static int rem(int a, int b) {
        if (b == 0)
            return a / 0; // defer to division by zero

        if (a < 0)
            return -rem(-a, b);

        if (b < 0)
            b = -b;

        if (b == 1)
            return 0;
        if (b == 2)
            return a & 1;

        while (a >= b)
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
