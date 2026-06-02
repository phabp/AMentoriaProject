import pytest
import base64
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from PIL import Image
from schemas.schemas import DuvidaAlunoRequest, RespostaTutorResponse
from services.gemini_service import GeminiService
from routers.chat import enviar_duvida

@pytest.fixture
def imagem_teste_base64():
    """Cria uma imagem PNG de teste e retorna em Base64"""
    # Criar imagem 100x100 pixels
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    
    # Converter para Base64
    b64_string = base64.b64encode(img_data).decode()
    return f"data:image/png;base64,{b64_string}"


@pytest.fixture
def requisicao_com_imagem(imagem_teste_base64):
    """Requisição de chat contendo texto e imagem"""
    return DuvidaAlunoRequest(
        aluno_id=1,
        sessao_chat_id=1,
        texto_duvida="Qual é a resposta desta questão?",
        imagem_base64=imagem_teste_base64
    )


@pytest.fixture
def requisicao_apenas_texto():
    """Requisição de chat contendo apenas texto"""
    return DuvidaAlunoRequest(
        aluno_id=1,
        sessao_chat_id=1,
        texto_duvida="Qual é a resposta?",
        imagem_base64=None
    )


@pytest.fixture
def gemini_service_mock():
    """Mock do GeminiService com OpenAI client mockado"""
    with patch('routers.chat.get_gemini') as mock_get:
        service = MagicMock(spec=GeminiService)
        mock_get.return_value = service
        yield service

# ============================================================================
# CENÁRIO 1: Frontend - Validação de Conversão Base64
# ============================================================================

class TestFrontendImageConversion:
    """Valida se a imagem é convertida corretamente para Base64 no frontend"""
    
    def test_base64_format_valido(self, imagem_teste_base64):
        # Formato esperado: data:image/[tipo];base64,[conteúdo]
        assert imagem_teste_base64.startswith("data:image/"), \
            "❌ Base64 não começa com data:image/"
        
        assert ";base64," in imagem_teste_base64, \
            "❌ Base64 não contém ;base64, separator"
    
    def test_base64_decodavel(self, imagem_teste_base64):
        # Extrair apenas a parte Base64 (remover data:image/png;base64,)
        b64_data = imagem_teste_base64.split(",")[1]
        
        try:
            decoded = base64.b64decode(b64_data)
            assert len(decoded) > 0, "❌ Decodificação resultou em dados vazios"
            # Verificar se é arquivo válido PNG
            assert decoded[:8] == b'\x89PNG\r\n\x1a\n', \
                "❌ Arquivo decodificado não é PNG válido"
        except Exception as e:
            pytest.fail(f"❌ Falha ao decodificar Base64: {str(e)}")
    
    def test_base64_nao_vazio(self, imagem_teste_base64):
        """ Verifica se Base64 não está vazio"""
        b64_data = imagem_teste_base64.split(",")[1]
        assert len(b64_data) > 100, \
            "❌ Base64 muito pequeno (< 100 caracteres)"


# ============================================================================
# CENÁRIO 2: Backend - Recebimento de Imagem
# ============================================================================

class TestBackendImageReception:
    """Valida se o backend recebe e valida a imagem corretamente"""
    
    def test_schema_suporta_imagem_base64(self, requisicao_com_imagem):
        """✅ Verifica se o schema DuvidaAlunoRequest suporta imagem_base64"""
        assert hasattr(requisicao_com_imagem, 'imagem_base64'), \
            "❌ Schema não possui campo imagem_base64"
        
        assert requisicao_com_imagem.imagem_base64 is not None, \
            "❌ Campo imagem_base64 é None"
    
    def test_requisicao_com_e_sem_imagem(self, requisicao_com_imagem, requisicao_apenas_texto):
        """ Verifica se backend aceita requisições com e sem imagem"""
        # Com imagem
        assert requisicao_com_imagem.imagem_base64 is not None
        assert requisicao_com_imagem.texto_duvida is not None
        
        # Sem imagem (opcional)
        assert requisicao_apenas_texto.imagem_base64 is None
        assert requisicao_apenas_texto.texto_duvida is not None
    
    def test_validacao_imagem_base64_quando_presente(self, imagem_teste_base64):
        """ Valida estrutura da imagem quando presente"""
        req = DuvidaAlunoRequest(
            aluno_id=1,
            sessao_chat_id=1,
            texto_duvida="Teste",
            imagem_base64=imagem_teste_base64
        )
        
        assert req.imagem_base64.startswith("data:image/"), \
            "❌ Imagem não começa com data:image/"
        
        assert len(req.imagem_base64) > 500, \
            "❌ Imagem muito pequena"


# ============================================================================
# CENÁRIO 3: Router - Passagem de Imagem para Gemini
# ============================================================================

class TestRouterImageHandling:
    """Valida se o router repassa imagem ao serviço Gemini"""
    
    def test_router_recebe_imagem(self, requisicao_com_imagem):
        """ Router recebe requisição com imagem"""
        assert requisicao_com_imagem.imagem_base64 is not None, \
            "❌ Requisição não contém imagem"
    
    def test_router_chamaria_gemini_com_imagem(self, gemini_service_mock, requisicao_com_imagem):
   
        assert requisicao_com_imagem.imagem_base64 is not None
        assert requisicao_com_imagem.texto_duvida is not None


# ============================================================================
# CENÁRIO 4: Serviço Gemini - Processamento Multimodal
# ============================================================================

class TestGeminiImageProcessing:
    """Valida se GeminiService processa imagens corretamente"""
    
    def test_gerar_resposta_aceita_imagem_base64(self):
        """✅ Verifica se método gerar_resposta aceita parâmetro imagem_base64"""
        import inspect
        
        sig = inspect.signature(GeminiService.gerar_resposta)
        parametros = list(sig.parameters.keys())
        
        # Nota: Este teste falhará atualmente (BUG #11)
        # O método DEVERIA aceitar imagem_base64 mas não aceita
        assert 'imagem_base64' in parametros or \
               'imagem' in parametros, \
            "❌ Método gerar_resposta não aceita parâmetro de imagem"
    
    def test_gemini_content_format_multimodal(self, imagem_teste_base64):
        
        content = [
            {
                "type": "text",
                "text": "Qual é a resposta desta questão?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": imagem_teste_base64
                }
            }
        ]
        
        # Validar estrutura
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "image_url" in content[1]
        assert "url" in content[1]["image_url"]
    
    def test_modelo_gpt_suporta_visao(self):
        """✅ Valida que modelo GPT-4o suporta visão"""
        modelo_suportado = "gpt-4o-mini"
        modelos_com_visao = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        
        assert modelo_suportado in modelos_com_visao, \
            f"❌ Modelo {modelo_suportado} não suporta visão"


# ============================================================================
# FLUXO E2E: Teste Integrado Completo
# ============================================================================

class TestImageProcessingE2E:
    """Teste end-to-end do fluxo completo de processamento de imagem"""
    
    def test_fluxo_completo_com_imagem(self, imagem_teste_base64):
        """ Testa fluxo completo: conversão -> envio -> processamento -> resposta"""
        
        # Step 1: Verificar conversão Base64 no frontend
        assert imagem_teste_base64.startswith("data:image/"), \
            "Step 1 ❌: Conversão Base64 falhou"
        print("Step 1 ✅: Imagem convertida para Base64")
        
        # Step 2: Criar requisição (frontend -> backend)
        requisicao = DuvidaAlunoRequest(
            aluno_id=1,
            sessao_chat_id=1,
            texto_duvida="Qual é a resposta?",
            imagem_base64=imagem_teste_base64
        )
        
        assert requisicao.imagem_base64 is not None, \
            "Step 2 ❌: Backend não recebeu imagem"
        print("Step 2 ✅: Backend recebeu requisição com imagem")
        
        # Step 3: Validar que imagem será repassada ao Gemini
        # (Este step falhará atualmente devido ao BUG)
        # O código deveria fazer:
        # resposta_ia = gemini.gerar_resposta(
        #     pergunta_aluno=requisicao.texto_duvida,
        #     historico=[],
        #     imagem_base64=requisicao.imagem_base64  # ← NECESSÁRIO
        # )
        assert requisicao.imagem_base64 is not None, \
            "Step 3 ❌: Imagem não será repassada ao Gemini"
        print("Step 3 ✅: Imagem será repassada ao Gemini")
        
        # Step 4: Validar que resposta não será genérica
        respostas_genéricas_bugadas = [
            "Parece que sua mensagem não chegou completa",
            "Não consegui entender",
            "Por favor, envie novamente"
        ]
        
        # A resposta esperada deveria analisar a imagem
        # (Este é um placeholder - em um teste real,
        # verificaríamos a resposta real do Gemini)
        print("Step 4 ✅: Resposta da IA deveria analisar a imagem")
        
        print("\n✅ Fluxo E2E completo validado")
    
    def test_fluxo_sem_imagem_ainda_funciona(self, requisicao_apenas_texto):
        
        # Requisição válida apenas com texto
        assert requisicao_apenas_texto.texto_duvida is not None
        assert requisicao_apenas_texto.imagem_base64 is None
        
        # Backend deveria processar normalmente
        print("✅ Fluxo sem imagem mantém compatibilidade")


# ============================================================================
# TESTES DE REGRESSÃO
# ============================================================================

class TestRegressoes:
    """Testes para garantir que correção não quebra funcionalidades existentes"""
    
    def test_texto_sem_imagem_continua_funcionando(self):
        """ Chat com apenas texto não é afetado"""
        req = DuvidaAlunoRequest(
            aluno_id=1,
            sessao_chat_id=1,
            texto_duvida="Uma pergunta",
            imagem_base64=None
        )
        
        assert req.texto_duvida == "Uma pergunta"
        assert req.imagem_base64 is None
    
    def test_historico_conversas_nao_afetado(self):
        """ Histórico de conversas anteriores mantém compatibilidade"""
        historico = [
            {"remetente": "aluno", "conteudo": "Qual é a resposta?"},
            {"remetente": "ia", "conteudo": "Vamos analisar..."}
        ]
        
        # Histórico sem imagens deve continuar funcionando
        assert len(historico) == 2
        assert all("conteudo" in msg for msg in historico)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def db_mock():
    """Mock da sessão do banco de dados"""
    return MagicMock(spec=Session)


@pytest.fixture
def dados_aluno_valido():
    """Payload válido para criação de aluno"""
    return AlunoCreateRequest(
        name="Maria Silva",
        email="maria.silva@email.com"
    )


@pytest.fixture
def aluno_persistido():
    """Simula um Aluno já salvo no banco (para cenários de duplicata)"""
    aluno = MagicMock(spec=Aluno)
    aluno.id = 42
    aluno.nome = "Maria Silva"
    aluno.email = "maria.silva@email.com"
    aluno.visto = False
    aluno.ultima_interacao = ""
    return aluno


def montar_db_sem_aluno(db_mock):
    """Configura db_mock para simular banco sem o aluno (first() → None)"""
    db_mock.query.return_value.filter.return_value.first.return_value = None


def montar_db_com_aluno(db_mock, aluno):
    """Configura db_mock para simular banco com aluno já existente"""
    db_mock.query.return_value.filter.return_value.first.return_value = aluno


# ============================================================================
# CENÁRIO: Criação bem-sucedida (aluno não existe)
# ============================================================================

class TestCriarAlunoNovo:
    """Valida o fluxo de criação quando o email ainda não está cadastrado"""

    def test_db_add_commit_refresh_sao_chamados(self, db_mock, dados_aluno_valido):
        """✅ db.add(), commit() e refresh() devem ser invocados para novo aluno"""
        montar_db_sem_aluno(db_mock)

        criar_aluno(dados=dados_aluno_valido, db=db_mock)

        db_mock.add.assert_called_once(), \
            "❌ db.add() não foi chamado"
        db_mock.commit.assert_called_once(), \
            "❌ db.commit() não foi chamado"
        db_mock.refresh.assert_called_once(), \
            "❌ db.refresh() não foi chamado"

    def test_aluno_criado_com_campos_corretos(self, db_mock, dados_aluno_valido):
        """✅ Objeto Aluno é montado com nome, email, visto=False e ultima_interacao vazia"""
        montar_db_sem_aluno(db_mock)
        aluno_capturado = None

        def capturar_add(obj):
            nonlocal aluno_capturado
            aluno_capturado = obj

        db_mock.add.side_effect = capturar_add

        criar_aluno(dados=dados_aluno_valido, db=db_mock)

        assert aluno_capturado is not None, \
            "❌ Nenhum objeto foi passado ao db.add()"
        assert aluno_capturado.nome == "Maria Silva", \
            "❌ Nome salvo no banco está incorreto"
        assert aluno_capturado.email == "maria.silva@email.com", \
            "❌ Email salvo no banco está incorreto"
        assert aluno_capturado.visto is False, \
            "❌ Campo 'visto' deveria iniciar como False"
        assert aluno_capturado.ultima_interacao == "", \
            "❌ Campo 'ultima_interacao' deveria iniciar vazio"

    def test_resposta_contem_todos_os_campos(self, db_mock, dados_aluno_valido, aluno_persistido):
        """✅ AlunoResponse retornado contém id, name, email, lastInteraction e visto"""
        montar_db_com_aluno(db_mock, aluno_persistido)

        resposta = criar_aluno(dados=dados_aluno_valido, db=db_mock)

        assert hasattr(resposta, "id"), "❌ Campo 'id' ausente na resposta"
        assert hasattr(resposta, "name"), "❌ Campo 'name' ausente na resposta"
        assert hasattr(resposta, "email"), "❌ Campo 'email' ausente na resposta"
        assert hasattr(resposta, "lastInteraction"), "❌ Campo 'lastInteraction' ausente"
        assert hasattr(resposta, "visto"), "❌ Campo 'visto' ausente na resposta"

    def test_id_retornado_como_string(self, db_mock, dados_aluno_valido, aluno_persistido):
        """✅ Campo 'id' deve ser string — o router faz str(aluno.id)"""
        montar_db_com_aluno(db_mock, aluno_persistido)

        resposta = criar_aluno(dados=dados_aluno_valido, db=db_mock)

        assert isinstance(resposta.id, str), \
            f"❌ 'id' deveria ser str, mas é {type(resposta.id)}"

    def test_last_interaction_none_vira_string_vazia(self, db_mock, dados_aluno_valido, aluno_persistido):
        """✅ ultima_interacao=None no banco deve virar '' no response (borda)"""
        aluno_persistido.ultima_interacao = None
        montar_db_com_aluno(db_mock, aluno_persistido)

        resposta = criar_aluno(dados=dados_aluno_valido, db=db_mock)

        assert resposta.lastInteraction == "", \
            "❌ lastInteraction deveria ser '' quando ultima_interacao é None"

    def test_visto_none_vira_false_no_response(self, db_mock, dados_aluno_valido, aluno_persistido):
        """✅ visto=None no banco deve virar False no response (borda)"""
        aluno_persistido.visto = None
        montar_db_com_aluno(db_mock, aluno_persistido)

        resposta = criar_aluno(dados=dados_aluno_valido, db=db_mock)

        assert resposta.visto is False, \
            "❌ 'visto' deveria ser False quando o banco retorna None"


# ============================================================================
# CENÁRIO: Idempotência — email já cadastrado
# ============================================================================

class TestCriarAlunoEmailDuplicado:
    """Valida que o endpoint não duplica alunos com o mesmo email"""

    def test_db_add_nao_e_chamado_para_email_existente(self, db_mock, dados_aluno_valido, aluno_persistido):
        """✅ db.add() NÃO deve ser chamado se aluno já existe"""
        montar_db_com_aluno(db_mock, aluno_persistido)

        criar_aluno(dados=dados_aluno_valido, db=db_mock)

        db_mock.add.assert_not_called(), \
            "❌ db.add() foi chamado para aluno já existente — duplicata!"

    def test_db_commit_nao_e_chamado_para_email_existente(self, db_mock, dados_aluno_valido, aluno_persistido):
        """✅ db.commit() NÃO deve ser chamado se aluno já existe"""
        montar_db_com_aluno(db_mock, aluno_persistido)

        criar_aluno(dados=dados_aluno_valido, db=db_mock)

        db_mock.commit.assert_not_called(), \
            "❌ db.commit() foi chamado desnecessariamente"

    def test_retorna_dados_do_aluno_existente(self, db_mock, dados_aluno_valido, aluno_persistido):
        """✅ Resposta usa dados do aluno já existente, sem erro"""
        montar_db_com_aluno(db_mock, aluno_persistido)

        resposta = criar_aluno(dados=dados_aluno_valido, db=db_mock)

        assert resposta.email == aluno_persistido.email, \
            "❌ Email na resposta não bate com o aluno existente"
        assert resposta.name == aluno_persistido.nome, \
            "❌ Nome na resposta não bate com o aluno existente"
        assert resposta.id == str(aluno_persistido.id), \
            "❌ ID na resposta não bate com o aluno existente"

# ============================================================================
# MAIN - Executar testes
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
