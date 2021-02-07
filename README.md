# Football-Data-Predictions ⚽🔍

Python script that shows statistics and predictions about different European soccer leagues using pandas and some AI techniques.

## Run it 🚀

First, run <code>git clone</code> or dowload the project in any directory of your machine. The data files are extracted from https://www.football-data.co.uk/data.php. **Data is not updated automatically**, so if you want to get current predictions and statistics you will have to manually enter these files in the project. The same goes for leagues/seasons that are not included in the project db folder. The only condition is that once this operation is done, you update the following line of code in the **_main.py_** file and put the location of your new file:

> data = pd.read_csv('db/your_file.csv')

Once you do that, you should install some dependencies via _pip_, in the case that you haven't installed them yet:

```sh
pip install pandas
pip install tkinter
pip install xlsxwriter
pip install sklearn
```

Once this is done, run the following command and follow the instructions of the terminal according to your wishes:

```sh
python main.py
```

In Linux or MacOS, you might have to type instead:

```sh
python3 main.py
```

**Warning:** _This project has been fully developed in python 3.9, so its operation for previous versions is not guaranteed._

## For developers 💻

Most of the models used are based on the same pandas dataframe. This dataframe is made up of a series of rows, each with a series of attributes (columns). These attributes can be divided into two types:

-   **Attributes of the current match:** Used by the models to measure errors and check their effectiveness. They will be the attributes to predict. They are located in the first columns of the dataframe.
-   **Attributes before the game:** Data available from the teams before the start of the game. Used to try to predict the attributes of the previous type described. For example, if a team has many goals in favor before a match, it is likely that in that same match (same row) the attribute indicating the number of goals scored by that team in that match is also high.

Thus, dataframes are made up of this combination of matches and attributes. For the best visualization of the dataframes, they are exported in an external folder called _trainingData_. The absolute data appears in the output.csv file, however, for all the models a conversion of the attributes is carried out, taking the average of them as a function of the number of games played.

In addition, the first matches of the dataframe have to be eliminated, because the averages still give uncertain values. As in any automatic learning process, the dataframe will be divided into test and training files, being able to select the percentage of matches dedicated to the test.

## The Algorithms 🧠

### Random Forest

A number N of decision trees is generated. These trees are formed by a series of leaves or rules, and in their final leaf, they will launch a prediction. The final prediction will consist of the arithmetic mean of the predictions made by the N trees.

<img src="/screenshots/sc1.PNG" alt="Screenshot 1" width="50%" height="50%" />

### Multilayer Preceptron

It is one of the simplest neural network models, made up of an input layer, an output layer, and N hidden layers or intermediate layers. Each layer will have a certain number of neurons and connections, each of which has a specific weight.

<img src="/screenshots/sc2.png" alt="Screenshot 1" width="50%" height="50%" />

## Next updates (To do list) 🔜

-   Improve the flow of elections through the interface
-   More artificial intelligence algorithms
-   More options to print the statistics and sort the classifications by attributes such as goals, corners ...
-   Allow the user to select the parameters of certain models
-   Who knows, maybe a graphical interface?

## Want to collaborate? 🙋🏻

Feel free to improve and optimize the existing code. To contribute to the project, read the previous points carefully and do the next steps with the project:

1. Fork it
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Need help ❓

Feel free to contact the developer if you have any questions or suggestions about the project or how you can help with it.
