public class Expr {
    public static int expr(int i, int j) {
        do {
            i = (i * j) - (i * j) + 4;
        } while (i < j);
        return i;
    }
}
