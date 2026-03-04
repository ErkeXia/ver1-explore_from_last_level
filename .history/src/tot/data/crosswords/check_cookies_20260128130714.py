from http.cookiejar import MozillaCookieJar

cj = MozillaCookieJar()
cj.load("NYT_cookies.txt", ignore_discard=True, ignore_expires=True)

names = sorted({c.name for c in cj})
print("loaded cookies:", len(cj))
print("has NYT-S:", "NYT-S" in names)
