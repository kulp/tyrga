package tyrga;

public class Builtin {
    public static native long add(long a, long b);

    public static long sub(long a, long b) {
        return add(a, -b);
    }

    public static long neg(long a) {
        return ~a + 1;
    }

    public static int div(int a, int b) {
        if (b == 0)
            return Integer.MAX_VALUE;

        // Optimizations
        if (b == 1)
            return a;

        boolean flip = false;
        if (b > 0) {
            flip = !flip;
            b = -b;
        }

        // operate on negative numbers, because in two's-complement, those can
        // be larger in magnitude than positive numbers can
        if (a > 0) {
            int c = -div(-a, b);
            return flip ? -c : c;
        }

        int c = 0;
        for (; a <= b; a -= b)
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
