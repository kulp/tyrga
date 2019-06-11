package tyrga;

public class Builtin {
    public static int rem(int a, int b) {
        if (b == 0)
            return a / 0; // defer to division by zero

        if (a < 0) {
            if (b < 0) {
                return -rem(-a, -b);
            } else {
                return -rem(-a,  b);
            }
        } else if (b < 0) {
            return rem(a, -b);
        } else {
            if (b == 1)
                return 0;
            if (b == 2)
                return a & 1;

            while (a >= b)
                a -= b;
            return a;
        }
    }
}

