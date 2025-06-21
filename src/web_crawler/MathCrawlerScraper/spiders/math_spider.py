import scrapy
from pathlib import Path
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from web_crawler.MathCrawlerScraper.spiders.seeds import SUPERPROF_SEED
class MathSpider(CrawlSpider):
    name = "MathSpider"
    start_urls = SUPERPROF_SEED # Aqui esta el seed de la semillas con todos los enlaces de SuperProf, cualquier otro enlace lo pueden poner el seed
    allowed_domains = ["www.superprof.es", "superprof.es"] # Cualquier otro Dominio que encuentren de interes, lo ponen
    _follow_links = True
    custom_settings = {
        'DEPTH_LIMIT': 3, 
        'CONCURRENT_REQUESTS': 8,
        'USER_AGENT': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0', # Si les da palo este UserAgent, pidanle a ChatGPT que les de otro
        'DOWNLOAD_DELAY': 2,
        'ROBOTSTXT_OBEY': True
    }
    link_stractor = LinkExtractor()
    rules = (
        Rule(LinkExtractor(
            allow=('apuntes', 'diccionario') # Donde dice allow, es las palabras que ustedes quieran que aparezcan en el enlace
        ), callback='parse', follow=True),
    )
    output_dir = Path(".data/raw/math")
    async def start(self):
        for url in self.start_urls:
            print(f"Entrando en la pagina {url}")
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        sanitized_url_part = response.url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_").replace("?", "_").replace("=", "_").replace("&", "_")
        filename_base = sanitized_url_part[4:100]
        filepath = self.output_dir / f"{filename_base}.html"
        try:
            filepath.write_bytes(response.body)
            self.log(f"Contenido HTML guardado en {filepath}")
        except Exception as e:
            self.log(f"Error al guardar el archivo {filepath}: {e}")
        for link in self.link_stractor.extract_links(response):
            yield scrapy.Request(link.url, callback=self.parse) # Esta es la parte recursiva, que saca a los links de la cola