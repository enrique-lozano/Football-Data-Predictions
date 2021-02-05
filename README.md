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

## Extra sections

Here you can add all the sections that occur to you in relation to your project. Some examples are:

-   Usage example 💡
-   Where the data is stored? 🕵️
-   What is the X Algorithm? 🧠

## For developers 💻

In this section we will put the extra commands and tasks that people who want to collaborate in the project will have to perform. For example, installation of programs, packages, dependencies ...

```sh
make example
npm test
```

And later, install the package in https://example.com and run

```sh
make example2
```

## Next updates 🔜

(if necessary)

## Issues 🤕

(if necessary)

## Want to collaborate? 🙋🏻

Feel free to improve and optimize the existing code. To contribute to the project, read the previous points carefully and do the next steps with the project:

1. Fork it
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Need help ❓

Feel free to contact the developer if you have any questions or suggestions about the project or how you can help with it.

## Screenshots 📸

Some screenshots of the project working if necessary:

<img src="/screenshots/sc1.png" alt="Screenshot 1" width="50%" height="50%" />

## -----------EXTRA CONTENT---------

### Beautiful commits

Emojis make everything nice, so you just have to include one of these in each of the commits you make:

> Translation improvements 🔣

> Code optimizations 🛠️

> Improved functionality 🛠️

> Working in XXX 🛠️

> Functionality implemented ✔️

> Class/File added ✔️

> Design improvements ✏️

> App deploy updated 🚀

> New release 🚀
