public class Except {
    public static int sub(int i) throws ArithmeticException {
        return 10 / i;
    }
    public static int except(int i) {
        try {
            return sub(i);
        } catch (ArithmeticException e) {
            return 0;
        }
    }

    public static int lastly(int i) {
        try {
            return sub(i);
        } finally {
            return -1;
        }
    }

    public static int except_lastly(int i) {
        try {
            return sub(i);
        } catch (ArithmeticException e) {
            return 0;
        } finally {
            return -1;
        }
    }

    public static int except2_lastly(int i) {
        try {
            return sub(i);
        } catch (ArithmeticException e) {
            return 0;
        } catch (RuntimeException e) {
            return -2;
        } finally {
            return -1;
        }
    }
}
