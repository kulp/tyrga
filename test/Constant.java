public class Constant {
    public static int    constant_int()     { return 1; }
    public static long   constant_long()    { return 1; }
    public static float  constant_float()   { return 1; }
    public static double constant_double()  { return 1; }
    public static Object constant_Object()  { return null; }

    static final int large = 123_456;
    public static int    large_int()        { return large; }
    public static long   large_long()       { return large; }
    public static float  large_float()      { return large; }
    public static double large_double()     { return large; }
}
