public class Sieve {
    public static void primes(int[] result)
    {
        int count = result.length;
        if (count > 0)
            result[0] = 2; // seed the result array
        if (count > 1)
            result[1] = 3; // seed the result array

        for (int i = 2; i < count; i++) {
            for (int test = result[i - 1] + 2; /* no test */; test++) {
                boolean composite = false;
                for (int j = 0; !composite && j < i && result[j] * result[j] <= test; j++) {
                    if (test % result[j] == 0) {
                        composite = true;
                    }
                }
                if (!composite) {
                    result[i] = test;
                    break;
                }
            }
        }
    }
};

