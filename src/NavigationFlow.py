from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time, geckodriver_autoinstaller
from datetime import timedelta

class TestInteraction1():
  geckodriver_autoinstaller.install() 
  options = webdriver.FirefoxOptions() 
  options.add_argument("--enable-quic --origin-to-force-quic-on=192.168.56.101:443 https:192.168.56.101")
  driver = webdriver.Firefox(options=options)
  
  print("Starting...")
  start = time.time()
  print(start)
  driver.get("https://192.168.56.101/")
  driver.maximize_window()

  time.sleep(10)
  driver.find_element(By.LINK_TEXT, "Documentação").click()
  time.sleep(3)
  driver.execute_script("window.scrollBy(0, 700)")
  time.sleep(11)
  driver.find_element(By.LINK_TEXT, "Download Diagrama Final Version").click()
  time.sleep(8)
  driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Apresentação 1").click()
  time.sleep(60)
  driver.back()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Apresentação 2").click()
  time.sleep(73)
  driver.back()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Apresentação 3").click()
  time.sleep(64)
  driver.back()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Apresentação 4").click()
  time.sleep(78)
  driver.back()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Apresentação Final").click()
  time.sleep(90)
  driver.back()
  time.sleep(7)
  driver.find_element(By.LINK_TEXT, "Home").click()
  time.sleep(4)
  driver.find_element(By.LINK_TEXT, "Equipa").click()
  time.sleep(55)
  driver.find_element(By.LINK_TEXT, "Documentação").click()
  time.sleep(2)
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
  driver.find_element(By.LINK_TEXT, "Calendário e Tarefas").click()
  time.sleep(28)
  driver.back()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Diagrama de Gantt v1.0").click()
  time.sleep(33)
  driver.find_element(By.LINK_TEXT, "Diagrama de Gantt v2.0").click()
  time.sleep(41)
  driver.find_element(By.LINK_TEXT, "Diagrama de Gantt v3.0").click()
  time.sleep(38)
  driver.find_element(By.LINK_TEXT, "Diagrama de Gantt v4.0").click()
  time.sleep(50)
  driver.find_element(By.CSS_SELECTOR, ".container-fluid img").click()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Módulos").click()
  time.sleep(28)
  driver.find_element(By.LINK_TEXT, "Documentação").click()
  time.sleep(2)
  driver.find_element(By.LINK_TEXT, "Protótipo").click()
  time.sleep(65)
  driver.back()
  time.sleep(2)
  driver.find_element(By.LINK_TEXT, "Relatório Descritor").click()
  time.sleep(80)
  driver.back()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Relatório Final").click()
  time.sleep(115)
  driver.back()
  time.sleep(3)
  driver.find_element(By.LINK_TEXT, "Vídeo Protótipo").click()
  time.sleep(180)
  driver.back()
  time.sleep(2)
  driver.find_element(By.LINK_TEXT, "Vídeo Promocional").click()
  time.sleep(120)
  driver.back()
  time.sleep(2)
  driver.find_element(By.LINK_TEXT, "Registo").click()
  time.sleep(10)
  driver.close()
  end = time.time()
  print("End: ")
  print(end)
  print("Elapsed time: ")
  print(str(timedelta(seconds=(end-start))))


def main():
  TestInteraction1()


## CAPTURA 32MINS
## QUIC Packets: 16562
## TOTAL: 17120

## CAPTURA 35MINS
## QUIC Packets: 19955

## Captura c fluxo acima mais saltado
## 45mins
## QUIC Packets: 29385
