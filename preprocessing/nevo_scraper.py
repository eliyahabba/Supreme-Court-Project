from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By


# shouldn't change these:
domain = "https://www.nevo.co.il"
login_url = "https://www.nevo.co.il/Authentication/UserLogin.aspx"

## initialize parameters
URLS = [
    # "https://www.nevo.co.il/SearchResults.aspx?query=92ae7418-e229-4598-8a73-02a9a9ce2bde#/"  # עעמ, דנגץ
    # "https://www.nevo.co.il/SearchResults.aspx?query=33135724-7f38-4388-bc5b-bfcf9d335d8d#/"  # 2017-2023
    # "https://www.nevo.co.il/SearchResults.aspx?query=a2cfddc3-02c3-4b40-8a8d-2e41fd28a920#/"  # 2010-2016
    # "https://www.nevo.co.il/SearchResults.aspx?query=d36299e3-07cf-46b0-9ffc-44ce35fd5cbe#/"  # 1998-2009
    "https://www.nevo.co.il/SearchResults.aspx?query=b81fa134-e2ef-4bb2-a046-6957a802696b#/"  # 1948-1997
]
# URL = "https://www.nevo.co.il/SearchResults.aspx?query=8320af0a-681e-4bab-8c74-0f55755f68db#/"  # search result URL
driver_path = r"C:\Program Files\chromedriver_win32\chromedriver.exe"  # path to where Chromedriver was downloaded
folder_path = r"C:\Users\Micha\Documents\HUJI\FinalProject\Nevo" #path to folder where files will be downloaded
USER = 'micha.hashkes@mail.huji.ac.il'  # username
PASSWORD = ''  # password


def log_in():
    chrome = webdriver.ChromeOptions()
    chrome.add_experimental_option("prefs", {'download.default_directory': folder_path})
    driver = webdriver.Chrome(driver_path, options=chrome)
    driver.get(login_url)
    username = driver.find_element(By.ID, "ContentPlaceHolder1_LoginForm1_Login1_UserName")
    password = driver.find_element(By.ID, "ContentPlaceHolder1_LoginForm1_Login1_Password")
    username.send_keys(USER)
    password.send_keys(PASSWORD)
    driver.find_element(By.ID, "ContentPlaceHolder1_LoginForm1_Login1_LoginButton").click()
    return driver


def scrape_from_link(url, driver):
    print(f'starting to scrape {url}')
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html')
    textlinks = soup.find_all(class_="docLink")
    for link in textlinks:
        print(domain + link.get('href'))
        driver.get(domain + link.get('href'))
    print(f"done scraping from {url}")
    return url
    # check_next_page(url, driver)


def scrape(driver, url):
    url = scrape_from_link(url, driver)

    driver.find_element(By.ID, "ContentPlaceHolder1_SearchResultsTemplate1_Paging2_btnNext").click()
    while driver.current_url != url:
        url = scrape_from_link(driver.current_url, driver)
        driver.find_element(By.ID, "ContentPlaceHolder1_SearchResultsTemplate1_Paging2_btnNext").click()
    print("done scraping all pages")


# def check_next_page(url, driver):
#     driver.find_element(By.ID, "ContentPlaceHolder1_SearchResultsTemplate1_Paging2_btnNext").click()
#     if driver.current_url != url:
#         scrape_from_link(driver.current_url, driver)
#     else:
#         print("done scraping all pages")


def run_scraper():
    driver = log_in()
    for url in URLS:
        scrape(driver, url)
    time.sleep(10)
    driver.quit()


run_scraper()
