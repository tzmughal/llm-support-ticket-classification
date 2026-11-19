"""Generate a synthetic, seeded support-ticket dataset.

Every ticket is generated from hand-written templates with randomised
slots (names, order numbers, amounts, products, error codes, dates).
No real customer data is used. Seed 42 makes the output reproducible:
re-running this script regenerates the same files.
"""
import csv
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "data"

FIRST = ["Ayesha", "Bilal", "Fatima", "Hassan", "Maryam", "Omar", "Sara", "Usman",
         "Zainab", "Ali", "Hina", "Danish", "Noor", "Ahmed", "Kiran", "Tariq"]
PRODUCTS = ["wireless headphones", "laptop charger", "smart watch", "coffee maker",
            "running shoes", "bluetooth speaker", "phone case", "desk lamp",
            "backpack", "kitchen blender", "tablet", "gaming mouse"]
CITIES = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad", "Multan"]

TEMPLATES = {
    "Billing": [
        "I was charged twice for my order {order}, please refund the duplicate payment of Rs {amount}.",
        "My card was charged Rs {amount} but I never received an invoice for order {order}.",
        "The subscription renewed automatically and I was billed Rs {amount} without any reminder.",
        "I need a refund for order {order}, the amount of Rs {amount} was deducted twice.",
        "My payment failed but the money was still taken from my bank account for order {order}.",
        "The discount code did not apply at checkout, I was overcharged by Rs {amount} on order {order}.",
        "Please send me the invoice and receipt for my payment of Rs {amount} on order {order}.",
        "I was charged for a premium plan I cancelled last month, please reverse the Rs {amount} charge.",
        "The price shown at checkout was Rs {amount} but my statement shows a higher amount for order {order}.",
        "I paid for order {order} through my wallet, the payment is stuck and the order is still unpaid.",
        "Can you explain this Rs {amount} charge on my bill? I do not recognise this transaction.",
        "My refund for order {order} has not arrived after two weeks, the amount was Rs {amount}.",
        "The bank declined my card yet the order total of Rs {amount} was deducted anyway.",
        "I upgraded my plan by mistake, please downgrade it and refund the extra Rs {amount}.",
        "Why was I charged a late fee of Rs {amount}? I paid my bill before the due date.",
        "The invoice for order {order} shows the wrong billing address, please correct it and resend it.",
        "I was billed in dollars instead of rupees, please fix the currency on my invoice for order {order}.",
        "My coupon for Rs {amount} off was not honoured, I paid the full price for order {order}.",
        "Please cancel my subscription and stop the monthly billing of Rs {amount}.",
        "The payment link for order {order} expired, but Rs {amount} was still debited from my account.",
    ],
    "Technical": [
        "The app crashes every time I open the {product} settings page, error code {code}.",
        "The website keeps showing error {code} when I try to check out.",
        "The app freezes on the loading screen after the latest update, I cannot use it at all.",
        "My {product} will not connect to Bluetooth, it shows error code {code}.",
        "The search feature returns no results even though the {product} is in stock.",
        "Videos keep buffering and the app is very slow since yesterday, error {code} appears.",
        "I cannot upload a photo, the upload fails with error code {code} every time.",
        "The checkout button does nothing when I tap it, I have tried restarting the app twice.",
        "The app logs me out randomly and shows error {code} on the login screen.",
        "Notifications stopped working after the update, I get no alerts for order {order}.",
        "The page for the {product} loads with broken images and missing prices.",
        "My order tracking page shows error {code} and will not refresh.",
        "The app drains my battery and crashes in the background, error code {code}.",
        "I cannot add the {product} to my cart, the button is greyed out.",
        "The payment page times out with error {code} before I can complete the order.",
        "The website is down, I get a 500 error when I open the homepage.",
        "The filter on the search page does not work, it shows the same products every time.",
        "My {product} firmware update failed at 80 percent, now the device will not start, error {code}.",
        "The app shows the wrong language after the update and the settings will not save.",
        "Audio keeps cutting out on calls through the app, the sound is distorted and choppy.",
    ],
    "Account": [
        "I cannot log in to my account, it says my password is incorrect even after resetting it.",
        "Please help me reset my password, the reset link never arrives in my email inbox.",
        "I want to change the email address on my account from my old address to my new one.",
        "My account was locked after too many login attempts, please unlock it.",
        "I cannot verify my phone number, the verification code never arrives.",
        "Please close my account and delete my personal data permanently.",
        "My profile name is spelled wrong, please update it on my account.",
        "Two-factor authentication is not working, I cannot receive the login code on my phone.",
        "I created two accounts by mistake, please merge them into one account.",
        "My account shows an old address in {city}, please update my delivery address.",
        "I cannot change my username, the settings page gives an error every time I save.",
        "Please reactivate my account, it was deactivated when I changed my email.",
        "My order history is missing from my account, my past orders do not show up.",
        "I want to update my phone number on my account but the verification keeps failing.",
        "The account recovery email goes to an address I no longer have access to.",
        "Please remove the old payment card saved on my account.",
        "My account was hacked, someone changed my password and my email address.",
        "I cannot sign up, the registration form says my email is already in use.",
        "Please change my account from personal to business and update the company name.",
        "My loyalty points disappeared from my account after the last update.",
    ],
    "Delivery": [
        "My order {order} was supposed to arrive on {date} but it is still not delivered.",
        "The tracking for order {order} has not updated in five days, where is my parcel?",
        "My package arrived damaged, the {product} inside the box is broken.",
        "I received the wrong item, I ordered a {product} but got something else in order {order}.",
        "The courier delivered my parcel to the wrong address in {city}, please help.",
        "One item is missing from my delivery for order {order}, the box was already open.",
        "My order {order} shows delivered but I never received anything.",
        "The delivery was delayed by a week with no update, please check order {order}.",
        "I need to change the delivery address for order {order} before it ships.",
        "The parcel for order {order} is stuck at the warehouse in {city} for three days.",
        "My {product} arrived with a dented box and a scratched screen.",
        "The courier tried to deliver while I was away, how do I reschedule delivery for order {order}?",
        "My order {order} was split into two parcels and only one has arrived.",
        "The delivery rider called but never arrived, order {order} is still marked out for delivery.",
        "Please deliver my order {order} to my office in {city} instead of my home address.",
        "The tracking number for order {order} is invalid, it shows no information.",
        "My package was returned to the sender without any reason, order {order}.",
        "The delivery charge was higher than shown at checkout for order {order}.",
        "I ordered the {product} for a gift, it must arrive before {date}, please confirm delivery.",
        "The parcel arrived late and the {product} inside was the wrong colour.",
    ],
}


def make_ticket(rng: random.Random, label: str) -> str:
    template = rng.choice(TEMPLATES[label])
    return template.format(
        order=f"ORD-{rng.randint(10000, 99999)}",
        amount=rng.choice([499, 999, 1499, 1999, 2499, 3499, 4999, 7999, 12999]),
        product=rng.choice(PRODUCTS),
        code=f"E{rng.randint(100, 999)}",
        date=rng.choice(["Monday", "Tuesday", "Friday", "last week", "12 November", "18 November"]),
        city=rng.choice(CITIES),
    )


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for label, templates in TEMPLATES.items():
        for _ in range(250):  # 250 per class x 4 classes = 1000 tickets
            rows.append({"text": make_ticket(rng, label), "label": label})
    rng.shuffle(rows)

    df = pd.DataFrame(rows)
    train, test = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["label"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    df.to_csv(OUT / "tickets.csv", index=False)
    print(f"total={len(df)} train={len(train)} test={len(test)}")
    print(df["label"].value_counts().to_dict())
    print("train label counts:", train["label"].value_counts().to_dict())
    print("test label counts:", test["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
