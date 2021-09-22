import json
import os


class Historico:

    def __init__(self, arquivo):
        if not arquivo:
            raise ValueError
        else:
            self.arquivo = arquivo

        fp = open(arquivo, 'a+')
        if fp:
            fp.close()
        else:
            raise OSError("Error during file handling ({})".format(arquivo))


    def salvar_no_historico(self, resultado):
        resultado['arquivo'] = os.path.basename(resultado['arquivo'])
        with open(self.arquivo, "r") as fp:
            historico = json.load(fp)
        if len(historico) == 0:
            prox_id = 1
        else:
            prox_id = int(sorted(self.buscar_historico(), key=int)[-1])+1
        historico[prox_id] = resultado
        with open(self.arquivo, 'w+') as fp:
            json.dump(historico, fp, indent=4)

    def buscar_historico(self):
        with open(self.arquivo, "r") as fp:
            historico = json.load(fp)
        return historico

    def buscar_ultimo(self):
        historico = self.buscar_historico()
        keys = list(historico.keys())
        for i in range(len(keys)):
            keys[i] = int(keys[i])
        keys = sorted(keys)
        last_key = "{}".format(keys[-1])
        return historico[last_key]

    def exportar(self, destino):
        with open(self.arquivo, "r") as fp:
            historico = json.load(fp)
        with open(destino, 'w+') as fp:
            json.dump(historico, fp, indent=4)

    def importar(self, origem):
        with open(origem, "r") as fp:
            historico = json.load(fp)
        with open(self.arquivo, 'w+') as fp:
            json.dump(historico, fp, indent=4)