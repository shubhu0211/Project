create database project;
use project;
select * from accident;

#1)Showing those data of accident where the age group of the driver below 18 and having experience below 1 year.
select * from accident where Age_Group_of_driver in ('Under 18','18-30') and Driving_experience='Below 1yr';


/* 2)find values Days, Types of Junction, Age Group of driver, Gender of driver when giving data of Driving experience in ('Below 1yr','1-2yr')  
and Cause of accident after taking drugs or drinks */
select  Day_of_week, Types_of_Junction as Junction, Age_Group_of_driver as 'Age Group', Gender_of_driver as Gender, Driving_experience 
as Experience, Cause_of_accident as Reason from accident 
where Cause_of_accident in ('Driving carelessly','Driving under the influence of drugs','No distancing','With Train','Unknown',
'Fall from vehicles') and Driving_experience in ('Below 1yr','1-2yr') 
group by Types_of_Junction, Day_of_week, Age_Group_of_driver, Gender_of_driver, Vehicle_driver_relation, Driving_experience, Cause_of_accident;


#3) find the data of A_No, Day_of_week, Types_of_Junction, Age_Group_of_driver where experince below 2 years and collision of 
# Vehicle with vehicle collision, Collision with roadside objects.
select A_No, Day_of_week, Types_of_Junction as Junction, Age_Group_of_driver as 'Age Group', Driving_experience as Experience,
Type_of_collision as Collision from accident 
where Type_of_collision in ('Vehicle with vehicle collision','Collision with roadside objects')
and Driving_experience in ('Below 1yr','1-2yr') 
group by A_No,Day_of_week, Types_of_Junction, Age_Group_of_driver, Driving_experience, Type_of_collision;


#4) find data having environment (raining, raining and windy, cloudy) and accident cause is driver mistake while 
# changing lane and etc.
select A_No, Time, Vehicle_driver_relation, Weather_conditions as Environment, Cause_of_accident as 'Accident Reason' 
from accident where Gender_of_Driver in ('Male','Female','Unknown') and Day_of_week in ('Sunday','Saturday') 
and Weather_conditions in ('Raining','Raining and Windy','Cloudy','Other') and Cause_of_accident in 
('Moving Backward','Overtaking','Changing lane to the left','Changing lane to the right','Overloading','Other',
'No priority to vehicle','Moving Backward');


#5) write querty to find no, junction type, collision where experience below 1 year, Junctions are No junction, Y Shape, 
# Crossing and time between 18:00:00 to 24:00:00,
select A_No, Types_of_Junction as Junction, Type_of_collision as Collision, Driving_experience as Experience, Time from accident 
where Time between '18:00:00' and '24:00:00' and Driving_experience='Below 1yr'and Types_of_Junction in 
('No junction','Y Shape','Crossing') group by A_No, Types_of_Junction, Type_of_collision, Time order by Time asc;


#6) create view having Days, Junction type, Age Group of driver, Gender of driver ,driving experience below 2 years and 
# collision having Vehicle with vehicle collision, Collision with roadside objects, Collision with animals, Collision 
# with roadside-parked vehicle, Rollove, Unknown)
create view accident1 as(select Day_of_week, Types_of_Junction as Junction, Age_Group_of_driver as 'Age Group', 
Gender_of_driver as Gender, Driving_experience as Experience, Type_of_collision as Collision from accident 
where Type_of_collision in ('Vehicle with vehicle collision','Collision with roadside objects','Collision with animals',
'Collision with roadside-parked vehicles','Rollover','Unknown') and Driving_experience in ('Below 1yr','1-2yr') 
group by Types_of_Junction, Day_of_week, Age_Group_of_driver, Gender_of_driver, Vehicle_driver_relation, 
Driving_experience, Type_of_collision);
select * from accident1;


#7) find day wise total count of accidents happened on the crossing junction in dakrness
select distinct Day_of_week, Types_of_Junction, count(A_No) as 'Total Accident' from accident where 
Light_conditions='Darkness - lights lit' and Types_of_Junction='Crossing' group by  
Day_of_week, Types_of_Junction order by 'Total Accident';


#8) display the number of male gender in age group in 18-30 had a accident on one way .
select Age_group_of_driver, Gender_of_driver, count(Gender_of_driver) as Total FROM accident WHERE  
Age_group_of_driver= '18-30' and Gender_of_driver in ('male','female') and Lanes_or_Medians='One Way' 
group by  Age_group_of_driver, Gender_of_driver;


#9) showing the data of those accident where person died in accident is in Elementry school on the 
# day of friday to saturday.
select A_No as ANo, Day_of_week as Days, Educational_level as Education, Types_of_Junction as 
Junction from accident where Educational_level in (select Educational_level from accident
where Educational_level='Elementary school') having Day_of_week in ('Friday','Saturday','Sunday');


#10) display a query were the age group of drive is 18-30 and gender of driver is female.
select age_group_of_driver , gender_of_driver, count(age_group_of_driver)as total from accident
where age_group_of_driver='18-30' AND gender_of_driver='female'group by age_group_of_driver,gender_of_driver;


#11) create view as age showing day_of_week, age_group_of_driver, gender_of_driver having 
#age below 18
create view Age as
(select day_of_week as Day, age_group_of_driver as Age, Gender_of_driver as gender,
type_of_collision as Type, Vehicle_movement as movement 
from accident where age_group_of_driver > 18);
select * from Age;


#12) find the number of accidents in which the type of collision is 
#collision with animals
select count(type_of_collision) from accident 
where type_of_collision = 'collision with animals';


#13) find the time, gender, driving_experience whose accident took place during daylight and cause of accident is driving carelessly
select A_No, gender_of_driver, driving_experience,light_conditions,
cause_of_accident from accident 
where light_conditions = 'daylight' 
and cause_of_accident = 'Driving carelessly';


#14) find age, experience, cause of accident and type of collision where the vehicle was going straight and collided with pedestrians
select A_No,age_group_of_driver as Age, driving_experience as Exp, cause_of_accident as Cause,
type_of_collision as Types
from accident where vehicle_movement = 'going straight' 
and type_of_collision = 'collision with pedestrians'; 


#15) find the no of females whose accidents happenend between 1am to 10am 
select time as time_of_accident, age_group_of_driver as Age, driving_experience as Exp,
cause_of_accident as Cause, type_of_collision as Types,
gender_of_driver as gender 
from accident group by time, age_group_of_driver, driving_experience, cause_of_accident, 
type_of_collision, gender_of_driver
having time between '01.00.00' and '10.00.00' and gender_of_driver = 'female' order by time;


#16) find the number of accident done by male, female
select  gender_of_driver, (age_group_of_driver) as Age
from accident group by gender_of_driver, age_group_of_driver 
having gender_of_driver < 18 order by gender_of_driver;


#17) create a view of age group of female drivers along with there driving experience, type of collisiom, gender of driver
create view female as
(select age_group_of_driver as Age, driving_experience as Exp, cause_of_accident as Cause,
type_of_collision as Types, gender_of_driver as Gender 
from accident where age_group_of_driver in
(select age_group_of_driver from accidentnew group by age_group_of_driver having gender_of_driver = 'female'));
select * from female;


#18) find the number of females drivers whose accident took place
select count(gender_of_driver) as females from accident 
where gender_of_driver ='female';


#19) find the average and count of drivers having age between 18 to 30 
select avg(age_group_of_driver)as AVG, count(age_group_of_driver) as COUNT from accident 
where age_group_of_driver = '18-30';


#20) Write a query to find the number of accident occured on each day of the week.
select distinct day_of_week as Day, count(time)as Accidents from accident 
group by day_of_week order by day_of_week desc;


#21) display the count of male had an accident while going straight and weather was rainy 
select count(gender_of_driver) from accident WHERE vehicle_movement=
"going straight" and weather_conditions="raining";





