public class Fields {
    int numerosity;
    long length;
    boolean truthiness;
    short diminutiveness;
    static int static_numerosity;
    static long static_length;
    static boolean static_truthiness;
    static short static_diminutiveness;

    public void access_fields() {
        numerosity++;
        length--;
        truthiness = !truthiness;
        diminutiveness = (short)-diminutiveness;
    }

    public static void access_statics() {
        static_numerosity++;
        static_length--;
        static_truthiness = !static_truthiness;
        static_diminutiveness = (short)-static_diminutiveness;
    }
}
