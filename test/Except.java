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
}
