################################
# Create edx set, validation set
################################


# Note: this process could take a couple of minutes


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(data.table)
library(tidyverse)
library(caret)
library(knitr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)


ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))


movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]



# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")


# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Exploratory Data Analysis

#Firstly, let´s check the dimension (the Number of Rows and Columns) of the Dataset and its class:    

dim(edx)
class(edx)

#We can see this table in tidy format with thousands of rows and with six observations in the columns:

edx%>% as_tibble()

names(edx)

#The edx dataset provided the following information

#* `userId` contains unique user identifier.
#* `movieId` contains unique movie identifier.
#* `rating` represents user's rating for a movie.
#* `timestamp` shows the date and time of user's rating in timestamp-format. Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

#* `title` includes the title as well as the publishing year of the rated movie.
#* `genres` includes all movie related genres, seperated with the symbol "|".

# It´s important to notice that each line of this data represents one rating of one movie by one user.

#We can see the number of unique users that provided ratings and how many unique movies were rated:

edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#Let’s look at some of the general properties of the data to better understand the challenges.

#The first thing we notice is that some movies get rated more than others. Here is the distribution:

edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

#Our second observation is that some users are more active than others at rating movies:

edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

#Ratings are made on a 5-star scale, with half-star increments.
#No movies have a rating of 0. So movies are rated from 0.5 to 5.0 in 0.5 increments. Have a look at these analyzes: 

edx %>% filter(rating == 0) %>% tally()

#Here follows the distribution of Movie Ratings:

summary(edx$rating)

edx %>% group_by(rating) %>% summarize(count = n())

#Visually, the Movie Ratings distribution can be seen here:

edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()

# This discrete rating distribution is more understandable in this histogram:

hist(edx$rating,
     main="Rating distribution",
     xlab="Ratings",
     xlim=c(0,5),
     col="orange",
     freq=TRUE
)
axis(side=1, at=sort(unique(edx$rating)), labels=sort(unique(edx$rating)))

# To complete, here follows the rank the movies in order of number of ratings:

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# It is clear where the highest ratings are concentrated, between 3 and 4 stars.

boxplot(edx$rating,
        main = "Rating distribution",
        xlab = "Ratings",
        ylab = "",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = FALSE
)
axis(side=1, at=sort(unique(edx$rating)), labels=sort(unique(edx$rating)))


# Let´s plot the popular genres:

genres_df<- edx%>%separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n())

genres_df%>%ggplot(aes(x=genres, y=count))+geom_point()+theme(axis.text.x = element_text(angle = 90, vjust = 0.5))



# Create Test and Train Data Sets

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test <- edx[test_index,]

#To make sure I will not include users and movies in the test set that do not appear in the training set, I will remove these entries using the semi_join function:
test_set <- test %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- test %>% 
  anti_join(train_set, by = "movieId")

train_set <- rbind(train_set, removed)

# Baseline model

mu_hat <- mean(train_set$rating)
mu_hat

#If we predict all unknown ratings with μ_hat we obtain the following RMSE:
naive_rmse <- RMSE(mu_hat, test_set$rating)
naive_rmse

#From looking at the distribution of ratings, we can visualize that this is the standard deviation of that distribution. We get a RMSE of about 1. It´s not good yet. 
#For instance, a participating team of the Netflix grand prize, had to get an RMSE of about 0.857. So we can definitely do better!

#As we go along, we will be comparing different approaches. Let’s start by creating a results table with this naive approach:
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
rmse_results

# Modeling movie effects
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i_hat = mean(rating - mu_hat))

#movie ratings distribution with the movie bias:

movie_avgs %>% qplot(b_i_hat, geom ="histogram", bins = 10, data = ., xlab = "Movie bias", main = "Movie bias distribution", color = I("black"))

#predict movie ratings based on the movie bias:
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu_hat + b_i_hat) %>%
  .$pred

#RMSE_Movie_Effec
model_1_RMSE <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effect Model", RMSE = model_1_RMSE))
rmse_results %>% knitr::kable()

# User effect
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u_hat = mean(rating - mu_hat - b_i_hat))

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu_hat + b_i_hat + b_u_hat) %>%
  .$pred

model_2_RMSE <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User Effects Model", RMSE = model_2_RMSE))
rmse_results %>% knitr::kable()

# Genre effect

genre_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g_hat = sum(rating - mu_hat - b_i_hat - b_u_hat/n()))

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId")%>%
  left_join(user_avgs, by = "userId")%>%
  left_join(genre_avgs, by = "genres")%>%
  mutate(pred = mu_hat + b_i_hat + b_u_hat+b_g_hat) %>%
  .$pred

model_3_RMSE <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User+ Genre Effects Model", RMSE = model_3_RMSE))
rmse_results %>% knitr::kable()

#Let's explore more the extrems of the data
movie_titles <- train_set %>% 
  select(movieId, title) %>%
  distinct()

# Top 10 best movies according to our prediction


train_set %>% count(movieId) %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i_hat)) %>% 
  select(title, b_i_hat, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

#And the ten worst movies according to our predictions
train_set %>% count(movieId) %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i_hat) %>% 
  select(title, b_i_hat, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

#Regularization

# Cross validation to pick the minimum value of RMSE for bi

# Regularization for the movie effect.

lambdas <- seq(0, 3, 0.25)
mu <- mean(train_set$rating)
just_the_sum <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)  
optm_lambda<-lambdas[which.min(rmses)]

l <- optm_lambda

mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+l)) 

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred


model_4_RMSE <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Regularized Movie Effect Model", RMSE = model_4_RMSE))
rmse_results %>% knitr::kable()

# Regularization for the movie + user effect.

lambdas <- seq(2, 6, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

l <- lambdas[which.min(rmses)]


model_5_RMSE <- min(rmses)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Regularized Movie + User Effect Model", RMSE = model_5_RMSE))
rmse_results %>% knitr::kable()


# Regularization for the movie + user +genre effect.

lambdas <- seq(2, 6, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u- mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

l <- lambdas[which.min(rmses)]


model_6_RMSE <- min(rmses)
model_6_RMSE
rmse_results <- bind_rows(rmse_results, data_frame(method = "Regularized Movie + User + Genre Effect Model", RMSE = model_6_RMSE))
rmse_results %>% knitr::kable()

l <- bestlambda 

# Results

# Calculate the biases on the edx data set with best lambda.

mu <- mean(edx$rating)
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))
b_g <- edx %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u- mu)/(n()+l))

predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred


# Calculate the predicted values for the validation data set
predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred) 

Final_RMSE <- RMSE(predicted_ratings, validation$rating)
Final_RMSE


