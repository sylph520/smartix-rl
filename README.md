# SmartIX

**How to set up TPC-H Benchmark and train SmartIX**

I. **Requirements**
*   Ubuntu Linux updated (`$ sudo apt-get update`)
*   GCC (`$ sudo apt-get install gcc`)
*   make (`$ sudo apt-get install make`)
*   MySQL version 14.14 (see Section II)
*   TPC-H version: 2.18.0 (see Section III)
*   Python 3.6  (see Section V)

- `sudo apt install gcc make`

II. **Installing MySQL**

https://dev.mysql.com/doc/refman/8.0/en/server-system-variables.html#annotations:bUUqIJO4EeyyYA_P5KREfA
```
sudo debconf-set-selections <<< "mysql-server mysql-server/lowercase-table-names select Enabled"
sudo apt-get install mysql-server
sudo mysql_secure_installation
```

```
VALIDATE PASSWORD PLUGIN (...) n
New password: YOUR PASSWORD
Re-enter new password: YOUR PASSWORD
Remove anonymous users? y
Disallow root login remotely? n
Remove test database and access to it? n
Reload privilege tables now? y
```

```
$ sudo ufw allow mysql
$ sudo systemctl start mysql
$ sudo systemctl enable mysql
$ sudo systemctl status mysql
```

Press **Ctrl+C** to exit MySQL status.

1. Create a new user:

```
sudo mysql -u root -p
CREATE USER 'dbuser'@'localhost' IDENTIFIED BY 'dbuser';
GRANT ALL PRIVILEGES ON *.* TO 'dbuser'@'localhost' WITH GRANT OPTION;
CREATE USER 'dbuser'@'%' IDENTIFIED BY 'dbuser';
GRANT ALL PRIVILEGES ON *.* TO 'dbuser'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
EXIT;
```

2. Create the database:

```
mysql -u dbuser -p
CREATE DATABASE tpch;
EXIT;
```

III. **Using  TPC-H Tools**

3. Click at TPC-H source code, available at the [TPC website](http://www.tpc.org/tpc_documents_current_versions/current_specifications.asp).

4. You have to fill in a register, and then they will send you a unique link by e-mail, so you can download and unzip the tools.

5. Go to the folder named `dbgen`, where there are the `.c` files.

6. Edit the `makefile.suite` file to match what we are showing below. We will use MySQL, but we have to set it to Oracle as the database here.

```
################
## CHANGE NAME OF ANSI COMPILER HERE
################
CC  	 = gcc
# Current values for DATABASE are: INFORMIX, DB2, TDAT (Teradata)
#                            SQLSERVER, SYBASE, ORACLE, VECTORWISE
# Current values for MACHINE are:  ATT, DOS, HP, IBM, ICL, MVS,
#                              	SGI, SUN, U2200, VMS, LINUX, WIN32
# Current values for WORKLOAD are:  TPCH
DATABASE = ORACLE
MACHINE  = LINUX
WORKLOAD = TPCH
```

7. Compile using the following command:

```
$ make -f makefile.suite
```

8. It will generate several compiled files. Now, we have to generate the database files. We use a scale factor of 1GB:

```
$ ./dbgen -s 1
```

9. It will generate one file for each table, to be inserted in the database, in the format tbl_name.tbl“. 

10. Create the tables using the file `dss.ddl`:

```
$ mysql -u dbuser -pdbuser tpch < dss.ddl
```

11. Log in on MySQL, and import the data to the database (pay attention to replace _~/path/ _for the actual path where the folder is):

```
load data local infile '~/tpch-kit/dbgen/region.tbl' into table REGION fields terminated by '|' lines terminated by '\n';
load data local infile '~/tpch-kit/dbgen/nation.tbl' into table NATION fields terminated by '|' lines terminated by '\n';
load data local infile '~/tpch-kit/dbgen/customer.tbl' into table CUSTOMER fields terminated by '|' lines terminated by '\n';
load data local infile '~/tpch-kit/dbgen/supplier.tbl' into table SUPPLIER fields terminated by '|' lines terminated by '\n';
load data local infile '~/tpch-kit/dbgen/part.tbl' into table PART fields terminated by '|' lines terminated by '\n';
load data local infile '~/tpch-kit/dbgen/orders.tbl' into table ORDERS fields terminated by '|' lines terminated by '\n';
load data local infile '~/tpch-kit/dbgen/partsupp.tbl' into table PARTSUPP fields terminated by '|' lines terminated by '\n';
load data local infile '~/tpch-kit/dbgen/lineitem.tbl' into table LINEITEM fields terminated by '|' lines terminated by '\n';
exit;
```
***************** add pk and fk ****************
```
-- Sccsid:     @(#)dss.ri	2.1.8.1
-- tpch Benchmark Version 8.0

USE tpch;

-- ALTER TABLE tpch.REGION DROP PRIMARY KEY;
-- ALTER TABLE tpch.NATION DROP PRIMARY KEY;
-- ALTER TABLE tpch.PART DROP PRIMARY KEY;
-- ALTER TABLE tpch.SUPPLIER DROP PRIMARY KEY;
-- ALTER TABLE tpch.PARTSUPP DROP PRIMARY KEY;
-- ALTER TABLE tpch.ORDERS DROP PRIMARY KEY;
-- ALTER TABLE tpch.LINEITEM DROP PRIMARY KEY;
-- ALTER TABLE tpch.CUSTOMER DROP PRIMARY KEY;


-- For table REGION
ALTER TABLE tpch.REGION
ADD PRIMARY KEY (R_REGIONKEY);

-- For table NATION
ALTER TABLE tpch.NATION
ADD PRIMARY KEY (N_NATIONKEY);

ALTER TABLE tpch.NATION
ADD FOREIGN KEY NATION_FK1 (N_REGIONKEY) references 
tpch.REGION(R_REGIONKEY);

COMMIT WORK;

-- For table PART
ALTER TABLE tpch.PART
ADD PRIMARY KEY (P_PARTKEY);

COMMIT WORK;

-- For table SUPPLIER
ALTER TABLE tpch.SUPPLIER
ADD PRIMARY KEY (S_SUPPKEY);

ALTER TABLE tpch.SUPPLIER
ADD FOREIGN KEY SUPPLIER_FK1 (S_NATIONKEY) references 
tpch.NATION(N_NATIONKEY);

COMMIT WORK;

-- For table PARTSUPP
ALTER TABLE tpch.PARTSUPP
ADD PRIMARY KEY (PS_PARTKEY,PS_SUPPKEY);

COMMIT WORK;

-- For table CUSTOMER
ALTER TABLE tpch.CUSTOMER
ADD PRIMARY KEY (C_CUSTKEY);

ALTER TABLE tpch.CUSTOMER
ADD FOREIGN KEY CUSTOMER_FK1 (C_NATIONKEY) references 
tpch.NATION(N_NATIONKEY);

COMMIT WORK;

-- For table LINEITEM
ALTER TABLE tpch.LINEITEM
ADD PRIMARY KEY (L_ORDERKEY,L_LINENUMBER);

COMMIT WORK;

-- For table ORDERS
ALTER TABLE tpch.ORDERS
ADD PRIMARY KEY (O_ORDERKEY);

COMMIT WORK;

-- For table PARTSUPP
ALTER TABLE tpch.PARTSUPP
ADD FOREIGN KEY PARTSUPP_FK1 (PS_SUPPKEY) references 
tpch.SUPPLIER(S_SUPPKEY);

COMMIT WORK;

ALTER TABLE tpch.PARTSUPP
ADD FOREIGN KEY PARTSUPP_FK2 (PS_PARTKEY) references 
tpch.PART(P_PARTKEY);

COMMIT WORK;

-- For table ORDERS
ALTER TABLE tpch.ORDERS
ADD FOREIGN KEY ORDERS_FK1 (O_CUSTKEY) references 
tpch.CUSTOMER(C_CUSTKEY);

COMMIT WORK;

-- For table LINEITEM
ALTER TABLE tpch.LINEITEM
ADD FOREIGN KEY LINEITEM_FK1 (L_ORDERKEY) references 
tpch.ORDERS(O_ORDERKEY);

COMMIT WORK;

ALTER TABLE tpch.LINEITEM
ADD FOREIGN KEY LINEITEM_FK2 (L_PARTKEY,L_SUPPKEY) references 
        tpch.PARTSUPP(PS_PARTKEY,PS_SUPPKEY);

COMMIT WORK;
```

12. Check the amount of rows for each table: 
* REGION: 5
* NATION: 25
* CUSTOMER: 150000
* SUPPLIER: 10000
* PART: 200000
* ORDERS: 1500000
* PARTSUPP: 800000
* LINEITEM: 6001215

13. Generate the files to be used in refresh functions. It will generate files named `delete.[num]`, and inserts to `orders.[num]`, and `lineitem.[num]` tables. Create a folder named `1` (one) and move the files intto this folder. We do this in order to separate the refresh files created with respect to a scale factor of 1.

```
./dbgen -s 1 -U 10000
mkdir 1
mv delete.* 1
mv orders.* 1
mv lineitem.* 1

```

IV. **Configuring the TPC-H Benchmark**

We need to create procedures, tables, and views to run our experiments.

14. Download the folder from [this link](https://drive.google.com/drive/folders/1aLSock8cYm18ONIveTg4rdL4iD7xcRHg?usp=sharing) and save it to your home folder.

15. Create folder `/QSRF` at `/home/user/`, and move files to this folder (`/home/user/QSRF`).

16. The files contained in the folder should be:
* `README.txt`: order to execute the files
* `CREATE TEMPORARY TABLES.sql`: create the tables to be used in data insertion and deletion;
* `CREATE VIEWS.sh`:  create views to test database query perfomance;
* `CREATE_PROCEDURE_DELETE_REFRESH_FUNCTION.sql`: create procedure to delete data;
* `CREATE_PROCEDURE_INSERT_REFRESH_FUNCTION.sql`: create procedure to insert data;
* `CREATE_PROCEDURE_QUERY_STREAM.sql`: create procedure for query stream (sequential query in the 22 views we created).

17. Create temporary tables:

```
mysql -u dbuser -p tpch 
source ~/QSRF/CREATE TEMPORARY TABLES.sql;
exit;
```

18. Create views (check the path in the `.sh` file):

```
$ cd ~/QSRF
$ ./CREATE_VIEWS.sh
```

19. Create procedures:

```
mysql -u dbuser -p tpch < ~/QSRF/CREATE_PROCEDURE_DELETE_REFRESH_FUNCTION.sql;
mysql -u dbuser -p tpch < ~/QSRF/CREATE_PROCEDURE_INSERT_REFRESH_FUNCTION.sql;
mysql -u dbuser -p tpch < ~/QSRF/CREATE_PROCEDURE_QUERY_STREAM.sql;
```

20. Check if views and procedures are created:

```
mysql -u dbuser -p tpch 
use tpch;
show tables;
```
34 rows in set (0.00 sec)
```
show procedure status where db = 'tpch';

(...)

| tpch | DELETE_REFRESH_FUNCTION | PROCEDURE | dbuser@localhost | 
| tpch | INSERT_REFRESH_FUNCTION | PROCEDURE | dbuser@localhost | 
| tpch | QUERY_STREAM        	 | PROCEDURE | dbuser@localhost | 

(...)
```
3 rows in set (0.00 sec)
```
exit;
```

V. **Configuring the Python Environment**
*   Python 3.6 (`$ sudo apt-get install python3.6`)
*   pip (`$ sudo apt-get install python3-pip`)
*   unixodbc (`$ sudo apt-get install unixodbc-dev`)
*   pyodbc (`$ pip3 install --user pyodbc`)
*   mysql-connector-python (`$ pip3 install mysql-connector-python`)
`sudo apt install python3-pip unixodbc-dev mysql-connector-python`
`pip3 install pyodbc`

21. Register the driver at MySQL (based on [MySQL Documentation](https://dev.mysql.com/doc/connector-odbc/en/connector-odbc-installation-binary-unix-tarball.html)).

22. Download MySQL connector from [this link](https://drive.google.com/drive/folders/16CEvXOK0bW3ecsC5tVY1R33uYYymDm1D?usp=sharing).

23. Go to the folder you downloaded and copy as follows:

```
sudo cp bin/* -r /usr/local/bin
sudo cp lib/* -r /usr/local/lib
```

24. Register the UNICODE driver:

```
$ sudo myodbc-installer -a -d -n "MySQL ODBC 8.0 Driver" -t "Driver=/usr/local/lib/libmyodbc8w.so"
```

25. Register ANSI driver: 

```
$ sudo myodbc-installer -a -d -n "MySQL ODBC 8.0" -t "Driver=/usr/local/lib/libmyodbc8a.so"
```

VI. **Training SmartIX**

26. Download SmartIX source code (or clone the repository).

27. Configure the database connection string in the database.py class init method: put your user, password, and database name to the connection string variable.

28. The same has to be done in the TPCH.py class: put your database connection info to the DB_CONFIG constant, as well as setting the REFRESH_FILES_PATH constant to the path you generated the refresh files back in Step 13.

29. Then you can finally start training the agent by running:

```
$ python3 environment.py > training.log
```

30. Finally, you can view training data in the `data` folder.
