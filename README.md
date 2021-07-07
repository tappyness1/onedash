HERE >>>https://one-dash.herokuapp.com/<<<

## How to run 

For scraping - 

`
python -m src.main
`

This will start the scraping for the following:

- Scrape character appearances in all managa chapters
- Scrape character list - a list of all the characters in OP
- Character details

These will be scraped all the way to the end chapter as set in the cfg file.

Run preprocessing -

`
python -m src.preprocess
`

For dashboard - 

For dash (flask) app
`
python -m dashboard
`

For streamlit 

`
streamlit run st_app.py
`

For deploying to heroku app 

```
heroku plugins:install heroku-builds
heroku builds:create -a one-dash
```
