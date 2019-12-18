from pyflink.dataset import ExecutionEnvironment
from pyflink.table import TableConfig, DataTypes, BatchTableEnvironment
from pyflink.table.descriptors import Schema, OldCsv, FileSystem

exec_env = ExecutionEnvironment.get_execution_environment()
exec_env.set_parallelism(2)
t_config = TableConfig()
t_env = BatchTableEnvironment.create(exec_env, t_config)

filename = open('/home/chikako_takasaki/tmp/input.txt', 'r', encoding='utf-8')
content = filename.read()
#t_env.connect(FileSystem().path('/home/chikako_takasaki/tmp/scarlet_letter.txt')) \
#                          .with_format(OldCsv().line_delimiter(' ')
#                                               .field('word', DataTypes.STRING())) \
#                          .with_schema(Schema().field('word', DataTypes.STRING())) \
#                          .register_table_source('mySource')

t_env.connect(FileSystem().path('/home/chikako_takasaki/work/flink/output2')) \
                          .with_format(OldCsv().field_delimiter(' : ')
                                               .field('word', DataTypes.STRING())
                                               .field('count', DataTypes.BIGINT())) \
                          .with_schema(Schema().field('word', DataTypes.STRING())
                                               .field('count', DataTypes.BIGINT())) \
                          .register_table_sink('mySink')

elements = [(word, 1) for word in content.split(" ")]
t_env.from_elements(elements, ["word", "count"]) \
     .group_by('word') \
     .select('word, count(1)') \
     .insert_into('mySink') 

t_env.execute("tutorial_job")
