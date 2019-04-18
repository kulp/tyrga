public class Switch {
    public static int table(int i, int j) {
        switch (i) {
            case 1: return j;
            case 2: return i * j;
            case 3: return i;
            default: return 0;
        }
    }

    public static int lookup(int i, int j) {
        switch (i) {
            case 1  : return j;
            case 10 : return i * j;
            case 100: return i;
            default: return 0;
        }
    }
}
