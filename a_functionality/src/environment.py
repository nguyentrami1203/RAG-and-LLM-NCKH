import copy


class MiniAmazonEnvironment:

    def __init__(self, initial_state):

        self.initial_state = copy.deepcopy(
            initial_state
        )

        self.state = copy.deepcopy(
            initial_state
        )

        self.action_history = []


    def reset(self):

        self.state = copy.deepcopy(
            self.initial_state
        )

        self.action_history = []

        return self.state


    def record_action(
        self,
        action,
        target=None,
        value=None
    ):

        self.action_history.append({
            "action": action,
            "target": target,
            "value": value
        })


    def click(self, target):

        self.record_action(
            "click",
            target
        )

        if target == "sony-product-link":

            self.state["navigation"][
                "current_page"
            ] = "product.html"


        elif target == "add-keyboard-cart":

            self.state["cart"][
                "keyboard_quantity"
            ] += 1


        elif target == "add-to-cart":

            self.state["cart"][
                "sony_quantity"
            ] += 1


        elif target == "add-to-wishlist":

            self.state["wishlist"][
                "sony_in_wishlist"
            ] = True


        elif target == "remove-sony":

            self.state["cart"][
                "sony_quantity"
            ] = 0


        elif target == "checkout":

            self.state["cart"][
                "checkout_started"
            ] = True


        elif target == "remove-wishlist-item":

            self.state["wishlist"][
                "sony_in_wishlist"
            ] = False


    def select(self, target, value):

        self.record_action(
            "select",
            target,
            value
        )

        if target == "qty-sony":

            self.state["cart"][
                "sony_quantity"
            ] = int(value)


    def create_wishlist(self, name):

        self.record_action(
            "create_wishlist",
            "save-list",
            name
        )

        self.state["wishlist"][
            "lists"
        ].append(name)


    def add_address(
        self,
        name,
        street,
        city,
        state,
        zip_code
    ):

        address = {
            "name": name,
            "street": street,
            "city": city,
            "state": state,
            "zip": zip_code
        }

        self.record_action(
            "add_address",
            "save-address",
            address
        )

        self.state["account"][
            "addresses"
        ].append(address)


    def stop(self, message):

        self.record_action(
            "stop",
            value=message
        )


    def get_state(self):

        return copy.deepcopy(
            self.state
        )


    def get_history(self):

        return copy.deepcopy(
            self.action_history
        )