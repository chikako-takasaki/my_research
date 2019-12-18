import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.api.common.functions.FilterFunction;
import py4j.GatewayServer;

public class Example {
  public static void execute(String[] args) throws Exception {
    final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

    DataStream<Person> flintstones = env.fromElements(
                                        new Person("Fred", 35),
                                        new Person("Wilma", 35),
                                        new Person("Pebbles", 2));

    DataStream<Person> adults = flintstones.filter(new FilterFunction<Person>() {
                                      @Override
                                      public boolean filter(Person person) throws Exception {
                                        return person.age >= 18;
                                      }
                                });

    adults.print();

    env.execute();
  }

  public static class Person {
    public String name;
    public Integer age;
    public Person() {};

    public Person(String name, Integer age) {
      this.name = name;
      this.age = age;
    };

    public String toString() {
      return this.name.toString() + ": age " + this.age.toString();
    };
  }

    public static void main(String[] args) {
      // GatewayServer 経由で機能を提供する
      Example application = new Example();
      GatewayServer gateway = new GatewayServer(application);
      gateway.start();
      System.out.println("Starting server...");
    }
}
