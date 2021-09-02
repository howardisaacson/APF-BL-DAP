import sqlite3

connection = sqlite3.connect('database.db')


with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO test_table (adjective, animal) VALUES (?, ?)",
            ('Green', 'Porcupine')
            )

cur.execute("INSERT INTO test_table (adjective, animal) VALUES (?, ?)",
            ('Snazzy', 'Ctenophore')
            )

cur.execute("INSERT INTO test_table (adjective, animal) VALUES (?, ?)",
            ('Bemused', 'Sloth')
            )

connection.commit()
connection.close()