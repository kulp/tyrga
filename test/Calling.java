public class Calling {
    public static void  _0_0() {}
    public static void  _0_1(int a) {}
    public static void  _0_2(int a, int b) {}
    public static void  _0_3(int a, int b, int c) {}

    public static int   _1_0() { return 100; }
    public static int   _1_1(int a) { return a + 101; }
    public static int   _1_2(int a, int b) { return a + b + 102; }
    public static int   _1_3(int a, int b, int c) { return a + b + c + 103; }

    public static long  _2_0() { return 200; }
    public static long  _2_1(int a) { return 201; }
    public static long  _2_2(int a, int b) { return a + b + 202; }
    public static long  _2_3(int a, int b, int c) { return a + b + c + 203; }

    public static void consume(int a) {}
    public static void consume(long a) {}

    public static void call_all() {
        _0_0();
        _0_1(1);
        _0_2(1, 2);
        _0_3(1, 2, 3);

        consume(_1_0());
        consume(_1_1(101));
        consume(_1_2(101, 102));
        consume(_1_3(101, 102, 103));

        consume(_2_0());
        consume(_2_1(201));
        consume(_2_2(201, 202));
        consume(_2_3(201, 202, 203));
    }
}
