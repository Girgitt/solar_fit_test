# PV / MPPT MQTT Simulator

Wielostronicowa aplikacja Dash do symulacji modułu PV oraz obciążenia sterowanego PWM, przygotowana jako środowisko testowe dla zewnętrznego algorytmu MPPT.

## Co robi aplikacja

Aplikacja łączy trzy role:

1. **Konfigurator modelu** – na stronie `Configuration` można ustawić parametry modułu PV, zakres nasłonecznienia, model obciążenia halogenowego oraz konfigurację MQTT.
2. **Silnik symulacji** – w tle działa pętla obliczeniowa, która wyznacza punkt pracy wynikający z przecięcia charakterystyki PV i modelu obciążenia.
3. **Interfejs testowy** – na stronie `Simulation` można sterować nasłonecznieniem i obserwować krzywą IV, krzywą mocy oraz aktualny punkt pracy.

## Model modułu PV

Model PV wykorzystuje referencyjne parametry:

- `Isc_ref`
- `Voc_ref`
- `Imp_ref`
- `Vmp_ref`
- `G_ref`

Dla nasłonecznienia `G_ref` krzywa IV jest konstruowana tak, aby zadany punkt `Vmp_ref / Imp_ref` był rzeczywistym maksimum mocy. Użyta rodzina krzywych ma postać:

`I(V) = Isc * (1 - (V / Voc)^a)^b`

Parametry `a` i `b` są wyliczane automatycznie z podanego punktu MPP.

### Zmienność względem nasłonecznienia

Zaproponowane ujęcie zmienności krzywej IV:

- `Isc` skaluje się prawie liniowo z nasłonecznieniem,
- `Voc` zmienia się łagodniej – przez zależność logarytmiczną od `G / G_ref`,
- kształt znormalizowanej krzywej pozostaje spójny, więc MPP przesuwa się naturalnie wraz ze zmianą `Isc` i `Voc`.

To jest celowo prostszy model niż pełny model jedno-diodowy, ale do testów MPPT jest wygodny, stabilny numerycznie i łatwy do strojenia.

## Model obciążenia halogenowego

Obciążenie jest traktowane jako żarówka halogenowa sterowana PWM.

### Parametry wejściowe

- napięcie znamionowe `V_nom`
- moc znamionowa `P_nom`
- współczynnik `R_hot / R_cold`
- stała czasowa cieplna `tau`
- wykładnik mapowania mocy na stan nagrzania

### Założenia modelu

1. Opór gorącego włókna jest liczony jako:
   `R_hot = V_nom^2 / P_nom`
2. Opór zimnego włókna:
   `R_cold = R_hot / (R_hot / R_cold)`
3. Bieżący opór zależy od stanu nagrzania włókna.
4. PWM wpływa na **moc średnią**, a więc pośrednio na temperaturę włókna.
5. Temperatura nie zmienia się skokowo, tylko z bezwładnością zadaną przez `tau`.

Dzięki temu przy szybkiej zmianie `set_duty` obciążenie reaguje bardziej realistycznie niż zwykły statyczny rezystor.

## MQTT

Aplikacja domyślnie:

- **subskrybuje** topic `set_duty`,
- może **publikować** symulowany pomiar V/I na topic pomiarowy,
- opcjonalnie może też **subskrybować** topic pomiarowy, aby pokazać zewnętrzny pomiar na wykresie.

To zachowanie zostało zrobione celowo, bo w opisie wymagań topic pomiarowy pojawia się jednocześnie jako wejście i jako naturalny kanał wyjściowy do testów MPPT.

### Oczekiwany format pomiaru

```json
{
  "age_ms": 12332,
  "channel_1": {
    "v": 12.23,
    "i": 0.45
  }
}
```

Klucz `channel_1` jest konfigurowalny.

### Oczekiwany format `set_duty`

Obsługiwane są dwie postacie:

```text
0.42
```

albo

```json
{"set_duty": 0.42}
```

## Strony aplikacji

## 1. Configuration

Na tej stronie ustawiasz:

- broker MQTT i topici,
- zakres suwaka nasłonecznienia,
- referencyjną charakterystykę IV modułu,
- parametry żarówki halogenowej,
- zakres i wartość początkową PWM.

Konfiguracja jest zapisywana do pliku `pv_sim_config.json`.

## 2. Simulation

Na tej stronie dostępne są:

- suwak nasłonecznienia,
- wybór źródła `duty`:
  - z MQTT,
  - ręczne nadpisanie,
- reset nagrzania żarówki,
- wykres:
  - krzywej IV (`V` względem `I`),
  - krzywej mocy `P(I)`,
  - aktualnego punktu pracy,
  - opcjonalnie ostatniego zewnętrznego pomiaru z MQTT.

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uruchomienie

```bash
./run.sh
```

albo

```bash
python app.py
```

Aplikacja domyślnie startuje na:

```text
http://0.0.0.0:8050
```

## Pliki projektu

- `app.py` – główny plik Dash z zakładkami i routingiem
- `pages/configuration.py` – strona konfiguracji
- `pages/simulation.py` – strona symulacji
- `config_store.py` – trwały zapis konfiguracji JSON
- `models.py` – modele PV i obciążenia
- `simulation_engine.py` – pętla symulacyjna w tle
- `mqtt_service.py` – klient MQTT w tle
- `pv_sim_config.json` – generowany automatycznie plik konfiguracji

## Uwagi praktyczne

- Aplikacja została przygotowana jako **lekki i stabilny symulator do testów MPPT**, a nie jako pełny model fizyczny ogniwa.
- Model PV jest wystarczająco realistyczny do strojenia algorytmu `set_duty -> szukanie Pmax`.
- Model żarówki daje naturalną dynamikę obciążenia i lepiej oddaje zachowanie włókna niż zwykły rezystor.
- Jeśli chcesz, łatwo rozbudować model o:
  - temperaturę ogniwa PV,
  - kilka kanałów obciążenia,
  - publikację dodatkowych pól MQTT,
  - zapisywanie historii punktu pracy.
