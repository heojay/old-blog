---
title: "DVD Rental Data Analysis"
author: "Jaewon Heo"
date: "2019-12-14"
output:
  html_document:
    theme: sandstone 
    highlight: tango
    code_folding: hide
    number_section: true
    self_contained: true
editor_options: 
  chunk_output_type: console
---



# PostgreSQL 설치하기

PostgreSQL은 확장 가능성 및 표준 준수를 강조하는 객체-관계형 데이터베이스 관리 시스템(ORDBMS)의 하나이다. 데이터 분석에 앞서 우선 PostgreSQL [사이트](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)에 접속해서 운영체제에 맞는 파일을 다운로드 한다.

자세한 설치 방법은 [이곳](http://www.postgresqltutorial.com/install-postgresql/)에서 확인하자.

# DVD 대여 데이터베이스 설치하기

DVD 대여 데이터베이스는 다음 [링크](http://www.postgresqltutorial.com/postgresql-sample-database/)에서 다운로드 받을 수 있다. 그 이후에는 다음 절차를 따라간다.
  
1. pgAdmin 도구를 시작하고 PostgreSQL 서버에 연결한다.
2. PostgreSQL 서버를 오른쪽 버튼으로 클릭하고 Create Database를 선택, dvdrental 데이터베이스를 만든다.
2. 다음으로 dvdrental 데이터베이스를 마우스 오른쪽 버튼으로 클릭하고 Restore… 메뉴 항목을 선택한다.
4. 그런 다음 데이터베이스 파일 경로를 넣고 (예: c:/dvdrental/dvdrental.tar), 복원 버튼을 클릭한다.
5. 마지막으로, 오브젝트 브라우저 패널에서 dvdrental 데이터베이스를 공용 스키마 및 기타 데이터베이스 오브젝트의 테이블이 표시된다.


# DVD 대여 데이터베이스 ERD


{% highlight r %}
library(datamodelr)

sQuery <- dm_re_query("postgres")
dm_dvdrental <- dbGetQuery(con, sQuery) 

dm_dvdrental <- as.data_model(dm_dvdrental)
graph <- dm_create_graph(dm_dvdrental, rankdir = "RL")
dm_render_graph(graph)
{% endhighlight %}

![plot of chunk ERD](/assets/article_images/DVD_Rental_Data_Analysis/ERD-1.png)


# 6가지 문제와 Insight

## 장르별 수요와 총 매출

사용한 5개의 table의 관계는 다음과 같다.


{% highlight r %}
category_r <- tbl(con, "category") %>% collect()
film_category_r <- tbl(con, "film_category") %>% collect()
rental_r <- tbl(con, "rental") %>% collect()
inventory_r <- tbl(con, "inventory") %>% collect()
payment_r <- tbl(con, "payment") %>% collect()

Q1_model <- dm_from_data_frames(category_r, film_category_r, inventory_r, rental_r, payment_r)

Q1_model <- dm_add_references(
  Q1_model,
  
  category_r$category_id == film_category_r$category_id,
  film_category_r$film_id == inventory_r$film_id,
  inventory_r$inventory_id == rental_r$inventory_id,
  rental_r$customer_id == payment_r$customer_id

)

Q1_graph <- dm_create_graph(Q1_model, rankdir = "LR", col_attr = c("column"))
dm_render_graph(Q1_graph)
{% endhighlight %}

![plot of chunk Q1 Graph](/assets/article_images/DVD_Rental_Data_Analysis/Q1 Graph-1.png)

- inventory, rental를 join해서 영화별 수요를 알아낸다.
- 여기에 payment를 join하면 영화별 매출을 알 수 있다.
- 이를 category, film_category와 join해서 영화마다 장르를 붙이고, 장르로 묶어준다.


{% highlight r %}
query <- "WITH t1 AS (SELECT c.name AS Genre, count(r.customer_id) AS Total_rent_demand
            FROM category c
            JOIN film_category fc
            USING(category_id)
            JOIN inventory i
            USING(film_id)
            JOIN rental r
            USING(inventory_id)
            GROUP BY 1
            ORDER BY 2 DESC),
     t2 AS (SELECT c.name AS Genre, SUM(p.amount) AS Total_sales
            FROM category c
            JOIN film_category fc
            USING(category_id)
            JOIN inventory i
            USING(film_id)
            JOIN rental r
            USING(inventory_id)
            JOIN payment p
            USING(rental_id)
            GROUP BY 1
            ORDER BY 2 DESC)
            
SELECT t1.Genre, t1.Total_rent_demand, t2.Total_sales
FROM t1
JOIN t2
ON t1.Genre = t2.Genre;
"
count.p<- sqldf(query)
datatable(count.p)
{% endhighlight %}

![plot of chunk Q1](/assets/article_images/DVD_Rental_Data_Analysis/Q1-1.png)

### Insights


{% highlight r %}
ggplot(count.p, aes(x = reorder(genre, -total_sales), y = total_sales)) + geom_bar(stat='identity') + theme(axis.text.x=element_text(angle=45, hjust=1)) + xlab('Genre') + ylab('Total Sales')
{% endhighlight %}

![plot of chunk P1-1](/assets/article_images/DVD_Rental_Data_Analysis/P1-1-1.png)


{% highlight r %}
ggplot(count.p, aes(x = reorder(genre, -total_sales), y = total_rent_demand)) + geom_bar(stat='identity') + theme(axis.text.x=element_text(angle=45, hjust=1)) + xlab('Genre') + ylab('Total Rent Demand')
{% endhighlight %}

![plot of chunk P1-2](/assets/article_images/DVD_Rental_Data_Analysis/P1-2-1.png)

- 16개의 장르가 있다.
- Total Sales와 Total Rent Demand 순위가 일치하지는 않는다
- Sports는 두 항목 모두 1등이고, Music은 반대로 두 항목 모두 16등이다.

## 장르별 유저 수

사용한 4개의 table의 관계는 다음과 같다.


{% highlight r %}
Q2_model <- dm_from_data_frames(category_r, film_category_r, inventory_r, rental_r)

Q2_model <- dm_add_references(
  Q2_model,
  
  category_r$category_id == film_category_r$category_id,
  film_category_r$film_id == inventory_r$film_id,
  inventory_r$inventory_id == rental_r$inventory_id

)

Q2_graph <- dm_create_graph(Q2_model, rankdir = "LR", col_attr = c("column"))
dm_render_graph(Q2_graph)
{% endhighlight %}

![plot of chunk Q2 Graph](/assets/article_images/DVD_Rental_Data_Analysis/Q2 Graph-1.png)

- category, film_category, inventory를 join해서 Genre별 재고를 확인한다.
- 이를 rental과 join해서 장르마다 DISTINCT한 customer_id 수를 세어준다.


{% highlight r %}
query <- "SELECT c.name AS Genre, count(DISTINCT r.customer_id) AS distinct_users
FROM category c
JOIN film_category fc
USING(category_id)
JOIN inventory i
USING(film_id)
JOIN rental r
USING(inventory_id)
GROUP BY 1
ORDER BY 2 DESC;"

count.p <- sqldf(query)
datatable(count.p)
{% endhighlight %}

![plot of chunk Q2](/assets/article_images/DVD_Rental_Data_Analysis/Q2-1.png)

### Insights


{% highlight r %}
ggplot(count.p, aes(x = reorder(genre, -distinct_users), y = distinct_users)) + geom_bar(stat='identity') + theme(axis.text.x=element_text(angle=45, hjust=1)) + xlab('Genre') + ylab('Distinct Users')
{% endhighlight %}

![plot of chunk P2](/assets/article_images/DVD_Rental_Data_Analysis/P2-1.png)

- 1번 질문에서와 같이 Sports는 여기서도 1등을 차지했다.
- 꼴찌였던 Music은 이전과 다르게 한 순위 오른 모습이다. 비교적 유저들이 대여에 열정적이지 않다는 것을 의미한다.

## 장르별 평균 렌탈료

사용한 3개의 table의 관계는 다음과 같다.


{% highlight r %}
film_r <- tbl(con, "film") %>% collect()

Q3_model <- dm_from_data_frames(category_r, film_category_r, film_r)

Q3_model <- dm_add_references(
  Q3_model,
  
  category_r$category_id == film_category_r$category_id,
  film_category_r$film_id == film_r$film_id

)

Q3_graph <- dm_create_graph(Q3_model, rankdir = "LR", col_attr = c("column"))
dm_render_graph(Q3_graph)
{% endhighlight %}

![plot of chunk Q3 Graph](/assets/article_images/DVD_Rental_Data_Analysis/Q3 Graph-1.png)

- category, film_category, film을 join해서 장르별 rental_rate를 구한다.


{% highlight r %}
query <- "SELECT c.name AS genre, ROUND(AVG(f.rental_rate),2) AS average_rental_rate
FROM category c
JOIN film_category fc
USING(category_id)
JOIN film f
USING(film_id)
GROUP BY 1
ORDER BY 2 DESC;"

count.p <- sqldf(query)
datatable(count.p)
{% endhighlight %}

![plot of chunk Q3](/assets/article_images/DVD_Rental_Data_Analysis/Q3-1.png)

### Insights


{% highlight r %}
ggplot(count.p, aes(x = reorder(genre, -average_rental_rate), y = average_rental_rate)) + geom_bar(stat='identity') + theme(axis.text.x=element_text(angle=45, hjust=1)) + xlab('Genre') + ylab('Average Rental Rate')
{% endhighlight %}

![plot of chunk P3](/assets/article_images/DVD_Rental_Data_Analysis/P3-1.png)

- 수요-공급 곡선을 생각해보면 렌탈요금과 유저수는 반비례할 것이다.
- 이에 근거해 Travel의 Distinct User가 적은 이유는 높은 렌탈료 때문일 수 있겠다. Action의 인기가 많은 이유는 그 반대이고.
- 반면, Sports는 렌탈요금이 비쌈에도 불구하고 인기있는 장르로, 이 장르의 유저들의 충성도가 높다고 볼 수 있다. Music과는 정 반대이다.

## 반납 시기별 필름 수

사용한 3개의 table의 관계는 다음과 같다.


{% highlight r %}
film_r <- tbl(con, "film") %>% collect()

Q4_model <- dm_from_data_frames(film_r, inventory_r, rental_r)

Q4_model <- dm_add_references(
  Q4_model,
  
  film_r$film_id == inventory_r$film_id,
  inventory_r$inventory_id == rental_r$inventory_id

)

Q4_graph <- dm_create_graph(Q4_model, rankdir = "LR", col_attr = c("column"))
dm_render_graph(Q4_graph)
{% endhighlight %}

![plot of chunk Q4 Graph](/assets/article_images/DVD_Rental_Data_Analysis/Q4 Graph-1.png)

- 우선 rental의 return_date와 rental_date를 이용해 기간을 구한다.
- 그리고 film, inventory, rental을 join해서 film의 rental_duration과 위에서 구한 기간을 비교해서 return 상태별 film 수를 세어준다.


{% highlight r %}
query <- "WITH t1 AS (Select *, DATE_PART('day', return_date - rental_date) AS date_difference
            FROM rental),
t2 AS (SELECT rental_duration, date_difference,
              CASE
                WHEN rental_duration > date_difference THEN 'Returned early'
                WHEN rental_duration = date_difference THEN 'Returned on time'
                ELSE 'Returned late'
              END AS Return_Status
          FROM film f
          JOIN inventory i
          USING(film_id)
          JOIN t1
          USING (inventory_id))
SELECT Return_status, count(*) As total_no_of_films
FROM t2
GROUP BY 1
ORDER BY 2 DESC;"

count.p <- sqldf(query)
datatable(count.p)
{% endhighlight %}

![plot of chunk Q4](/assets/article_images/DVD_Rental_Data_Analysis/Q4-1.png)

### Insights


{% highlight r %}
ggplot(count.p, aes(x = return_status, y = total_no_of_films/sum(total_no_of_films))) + geom_bar(stat='identity') + xlab('Return Status') + ylab('Number of Films') + scale_y_continuous(labels = scales::percent_format(accuracy = 1))
{% endhighlight %}

![plot of chunk P4](/assets/article_images/DVD_Rental_Data_Analysis/P4-1.png)

- 48%가 조기반납, 11%가 정시반납, 그리고 41%가 연체한다는 것을 알 수 있다.

## Rent A Flim의 국가별 고객 수와 총 매출

사용한 5개의 table의 관계는 다음과 같다.


{% highlight r %}
country_r <- tbl(con, "country") %>% collect()
city_r <- tbl(con, "city") %>% collect()
address_r <- tbl(con, "address") %>% collect()
customer_r <- tbl(con, "customer") %>% collect()

Q5_model <- dm_from_data_frames(country_r, city_r, address_r, customer_r, payment_r)

Q5_model <- dm_add_references(
  Q5_model,
  
  country_r$country_id == city_r$country_id,
  city_r$city_id == address_r$city_id,
  address_r$address_id == customer_r$address_id,
  customer_r$customer_id == payment_r$customer_id


)

Q5_graph <- dm_create_graph(Q5_model, rankdir = "LR", col_attr = c("column"))
dm_render_graph(Q5_graph)
{% endhighlight %}

![plot of chunk Q5 Graph](/assets/article_images/DVD_Rental_Data_Analysis/Q5 Graph-1.png)

- country, city, address를 join해 해당 주소가 어느 국가인지 알아낸다.
- 이를 payment, customer를 join해 알아낸 고객별 지불 금액과 join해서 국가별로 고객이 사용한 금액을 구한다.


{% highlight r %}
query <- "SELECT country, count(DISTINCT customer_id) AS customer_base, SUM(amount) AS total_sales
FROM country
JOIN city
USING(country_id)
JOIN address
USING(city_id)
JOIN customer
USING (address_id)
JOIN payment
USING(customer_id)
GROUP BY 1
ORDER BY 2 DESC;"

count.p <- sqldf(query)
datatable(count.p)
{% endhighlight %}

![plot of chunk Q5](/assets/article_images/DVD_Rental_Data_Analysis/Q5-1.png)

### Insights

- Rent A Film은 108개 국가에 고객을 가지고 있다.
- Total Sales와 Customer base가 가장 많은 곳은 India이다. 역시 발리우드...
- Total Sales가 가장 작은 나라는 American Samoa로 47.85이다. Customer base가 1명인 나라는 총 41개국이다.

## Rent A Film 상위 5 고객 정보

사용한 5개의 table의 관계는 다음과 같다.


{% highlight r %}
Q6_model <- dm_from_data_frames(country_r, city_r, address_r, customer_r, payment_r)

Q6_model <- dm_add_references(
  Q6_model,
  
  country_r$country_id == city_r$country_id,
  city_r$city_id == address_r$city_id,
  address_r$address_id == customer_r$address_id,
  customer_r$customer_id == payment_r$customer_id


)

Q6_graph <- dm_create_graph(Q6_model, rankdir = "LR", col_attr = c("column"))
dm_render_graph(Q6_graph)
{% endhighlight %}

![plot of chunk Q6 Graph](/assets/article_images/DVD_Rental_Data_Analysis/Q6 Graph-1.png)

- 5번과 같은 방법으로 테이블을 join하고, customer 정보를 중심으로 출력한다.


{% highlight r %}
query <- "WITH t1 AS (SELECT *, first_name || ' ' || last_name AS name
		    FROM customer)
SELECT name, email, address, phone, city, country, sum(amount) AS total_purchase_in_currency
FROM t1
JOIN address
USING(address_id)
JOIN city
USING (city_id)
JOIN country
USING (country_id)
JOIN payment
USING(customer_id)
GROUP BY 1,2,3,4,5,6
ORDER BY 7 DESC
LIMIT 5;"

count.p <- sqldf(query)
datatable(count.p)
{% endhighlight %}

![plot of chunk Q6](/assets/article_images/DVD_Rental_Data_Analysis/Q6-1.png)

### Insights

- 5번 질문에서 국가 순위 1등이었던 인도가 1등부터 5등까지 중에 나타나지 않는다.
- 1등은 Rent A Film의 유일한 Runion 출신 사람이다. 그러나 이것이 나라 이름으로 보이지는 않는다.


# 결론

- 가격이 높아도 유저 수가 많은, 즉 충성도가 높은 장르인 Sports, Sci-Fi 장르의 영화를 지속적으로 확보할 필요가 있다.
- 41%의 고객이 연체한다는 점은 문제다. 연체료를 높이거나 하는 방법으로 이 부분을 개선할 필요가 있다.
