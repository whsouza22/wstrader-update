from iqoptionapi.stable_api import IQ_Option
import sys

print('\n')
print(f'╰──› Iniciando Conexão.')

email = ''
senha = ''
API = IQ_Option(email,senha)

    

### Função para conectar na IQOPTION ###
check, reason = API.connect()
if check:
    print(f'  ─› Conectado com sucesso')
else:
    if reason == '{"code":"invalid_credentials","message":"You entered the wrong credentials. Please ensure that your login/password is correct."}':
        print('❌Email ou senha incorreta')
        input('─› Aperte enter para sair')
        sys.exit()
        
    else:
        print('❌Houve um problema na conexão')
        print(reason)
        input('─› Aperte enter para sair')
        sys.exit()

print('\n\n  ─› Puxando dados dos ativos...\n\n')
pares = API.update_ACTIVES_OPCODE()

msg = 'ACTIVES = {\n'
for i in pares:
    msg+= f'     "{i}" : {pares[i]},\n'

msg += '    }' 
print(msg)

arquivo = open('constants.py', 'w')
arquivo.writelines(msg)
arquivo.close()


input('\n\nSeu arquivo constants.py foi gerado com sucesso. Basta substituir ele dentro da pasta da API.')
sys.exit()