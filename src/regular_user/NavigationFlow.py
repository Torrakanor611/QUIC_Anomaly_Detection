from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import time
import geckodriver_autoinstaller
from selenium.webdriver.common.keys import Keys

class TestInteraction1():
  geckodriver_autoinstaller.install() 
  options = webdriver.FirefoxOptions() 
  options.add_argument("--enable-quic --origin-to-force-quic-on=192.168.56.101:443 https:192.168.56.101")
  driver = webdriver.Firefox()
  print("Starting...")
  start = time.time()
  print(start)
  driver.get("https://192.168.56.101/")
  driver.maximize_window()

  time.sleep(10)
  driver.find_element(By.LINK_TEXT, "Equipa").click()
  time.sleep(15)
  driver.find_element(By.LINK_TEXT, "Módulos").click()
  time.sleep(28)
  driver.find_element(By.CSS_SELECTOR, ".container-fluid img").click()
  time.sleep(5)
  driver.find_element(By.LINK_TEXT, "Documentação").click()
  time.sleep(12)
  driver.find_element(By.LINK_TEXT, "Relatório Descritor").click()
  time.sleep(80)
  driver.back()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Protótipo").click()
  time.sleep(65)
  driver.back()
  time.sleep(2)
  driver.find_element(By.LINK_TEXT, "Vídeo Promocional").click()
  time.sleep(120)
  driver.back()
  time.sleep(2)
  driver.find_element(By.LINK_TEXT, "Vídeo Protótipo").click()
  time.sleep(180)
  driver.back()
  for i in range(2):
    driver.execute_script("window.scrollBy(0, 530)")
    time.sleep(6)
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Registo Semanal").click()

  for i in range(33):
    driver.execute_script("window.scrollBy(0, 300)")
    time.sleep(40)

  driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
  time.sleep(5)
  driver.back()
  time.sleep(8)
  driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
  time.sleep(10)
  driver.find_element(By.LINK_TEXT, "Diagrama de Gantt v1.0").click()
  time.sleep(35)
  driver.find_element(By.LINK_TEXT, "Home").click()
  time.sleep(10)
  driver.close()
  end = time.time()
  print("End: ")
  print(end)
  print("Elapsed time: ")
  print(str(timedelta(seconds=(end-start))))


def main():
  TestInteraction1()


##PACKETS QUIC COMPLETE -> 20506
## TOTAL: 20655

## CAPTURA 32MINS
## FLOW acima descrito
## QUIC Packets: 16562
## TOTAL: 17120