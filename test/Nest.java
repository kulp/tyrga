public class Nest {
    public static int nest(int i, int j) {
        int l = 1;
        {
            int k = 1;
            while (j < k) {
                i = i + k;
                j = j - i;
                for (int m = 0; m < j; m++) {
                    i = i + j;
                    k = k - i;
                    do {
                        i = k * m;
                        j--;
                    } while (i < j);
                }
            }
        }
        return l;
    }
}
