public class Deep {
    public long sum = 0;

    public static int add(int a, int b) {
        return a + b;
    }

    public long calling() {
        return add(0,add(1,add(2,add(3,add(4,add(5,add(6,add(7,add(8,add(9,add(10,add(11,add(12,add(13,add(14,add(15,add(16,17)))))))))))))))));
    }
}
