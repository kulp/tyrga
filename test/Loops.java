public class Loops {
    public static int deep5() {
        int sum = 0;
        final int max = 3;
        for (int i = 0; i < max; i++)
        for (int j = 0; j < max; j++)
        for (int k = 0; k < max; k++)
        for (int l = 0; l < max; l++)
        for (int m = 0; m < max; m++)
            sum += i + j + k + l + m;

        return sum;
    }
}
