import scrapy
import re
from pathlib import Path
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from googlesearch import search

class DispatchSpider(CrawlSpider):
    name = "DispatchSpider"
    start_urls = []
    allowed_domains = [ "www.vitutor.com",
                        "www.unprofesor.com",
                        "www.matesfacil.com",
                        "www.disfrutalasmatematicas.com",
                        "www.areamatematica.com",
                        "www.smartick.es/blog",
                        "www.superprof.es/apuntes",
                        "www.educapeques.com/recursos-para-el-aula/matematicas",
                        "www.profedematematicas.es",
                        "www.elgimnasiodelamente.com/matematicas",
                        "www.ejerciciosmatematicas.net",
                        "es.khanacademy.org/math",
                        "www.mundoprimaria.com/recursos-educativos/matematicas"
                    ]
    _follow_links = True
    custom_settings = {
        'DEPTH_LIMIT': 1,
        'CONCURRENT_REQUESTS': 8,
        'USER_AGENT': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0',
        'DOWNLOAD_DELAY': 2,
        'ROBOTSTXT_OBEY': True
    }
    link_stractor = LinkExtractor()
    rules = (
        Rule(LinkExtractor(
            deny=('video', 'adds', 'add', ''),
            deny_domains=('youtube.com', 'facebook.com', 'instagram.com', 'tiktok.com')
        ), callback='parse', follow=True),
    )
    output_dir = Path(".data/raw/math")

    def __init__(self, query=None, *a, **kw):
        super().__init__(*a, **kw)
        if query:
            self.start_urls = self._get_init_urls()
            
    async def start(self):
        for url in self.start_urls:
            print(f"Entrando en la pagina {url}")
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        sanitized_url_part = response.url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_").replace("?", "_").replace("=", "_").replace("&", "_")
        filename_base = sanitized_url_part[:100]
        filepath = self.output_dir / f"{filename_base}.txt"
        try:
            filepath.write_bytes(response.body)
            self.log(f"Contenido HTML guardado en {filepath}")
        except Exception as e:
            self.log(f"Error al guardar el archivo {filepath}: {e}")
        for link in self.link_stractor.extract_links(response):
            yield scrapy.Request(link.url, callback=self.parse)

    def _get_init_urls(self, query: str) -> list[str]: # HAce la busqueda en internet, dada una query
        links = []
        for domain in self.allowed_domains:
            domain = domain[:4] #quitar el www
            item = search(term=f"{domain} {query}", num_results=3, lang='es')
            links += item
        return links